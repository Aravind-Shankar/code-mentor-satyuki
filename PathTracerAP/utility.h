#pragma once
#define GLM_FORCE_CUDA
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/random.h>
#include <thrust/remove.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define ABS(x) ((x) < 0 ? -(x) : (x))
#define IS_EQUAL(x, y) (ABS((x) - (y)) < EPSILON)
#define IS_LESS_THAN(x, y) ((x) < (y) - EPSILON)
#define IS_MORE_THAN(x, y) ((x) > (y) + EPSILON)
#define CLAMP(value, min, max) ((value) < (min) ? (min) : ((value) > (max) ? (max) : (value)))
#define CEIL(x,y) (((x) + (y) - 1) / (y))
#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f


void printCUDAMemoryInfo()
{
    size_t free_bytes, total_bytes;

    cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (cuda_status != cudaSuccess) 
    {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(cuda_status) << std::endl;
        return;
    }

    // Print the memory information
    std::cout << "Free GPU memory: " << free_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Total GPU memory: " << total_bytes / (1024.0 * 1024.0) << " MB" << std::endl;

}

__inline__ __host__ __device__
inline unsigned int utilHash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__inline__ __host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilHash((1 << 31) | (depth << 22) | iter) ^ utilHash(index);
    return thrust::default_random_engine(h);
}

__inline__ __host__ __device__ glm::vec3 reflectRay(const glm::vec3& incident, const glm::vec3& normal)
{
    return glm::normalize(incident - normal * 2.0f * glm::dot(incident, normal));
}

__inline__ __host__ __device__ glm::vec3 transformDirection(glm::vec3& direction, const glm::mat4& matrix)
{
    return glm::vec3(matrix * glm::vec4(direction, 0.0f));
}


__inline__ __host__ __device__ glm::vec3 transformPosition(const glm::vec3& position, const glm::mat4& matrix)
{
    return glm::vec3(matrix * glm::vec4(position, 1.0f));
}

__inline__ __host__ __device__ glm::vec3 transformNormal(glm::vec3& normal, const glm::mat4& matrix)
{
    glm::mat3 upper_left_matrix = glm::mat3(matrix);
    glm::mat3 inverse_transpose = glm::transpose(glm::inverse(upper_left_matrix));

    return inverse_transpose * normal;
}

__forceinline__ __host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float cos_theta = sqrt(u01(rng)); // cos(theta)
    float sin_theta = sqrt(1 - cos_theta * cos_theta); // sin(theta)
    float phi = u01(rng) * TWO_PI;

    glm::vec3 u = glm::normalize(normal);
    glm::vec3 v = glm::normalize(glm::cross((ABS(u.x) > .1 ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)), u));
    glm::vec3 w = glm::cross(u, v);

    return cos_theta * u + cos(phi) * sin_theta * v + sin(phi) * sin_theta * w;
}


float fresnel_reflectance(glm::vec3 incident_direction, glm::vec3 surface_normal, float refractive_index) {
    // Ensure that incident_direction and surface_normal are normalized
    incident_direction = glm::normalize(incident_direction);
    surface_normal = glm::normalize(surface_normal);

    // Calculate the cosine of the angle between incident direction and surface normal
    float cos_theta_i = glm::dot(incident_direction, surface_normal);

    // Check if the ray is coming from the other side of the surface
    if (cos_theta_i < 0) {
        cos_theta_i = -cos_theta_i;
        surface_normal = -surface_normal;
    }

    // Calculate the sine of the angle between incident direction and surface normal
    float sin_theta_i = std::sqrt(1.0f - cos_theta_i * cos_theta_i);

    // Calculate the sine of the angle in the refracted medium using Snell's Law
    float sin_theta_t = refractive_index * sin_theta_i;

    // Check if total internal reflection occurs
    if (sin_theta_t >= 1.0f) {
        return 1.0f;  // Total internal reflection, fully reflective
    }

    // Calculate the cosine of the angle in the refracted medium
    float cos_theta_t = std::sqrt(1.0f - sin_theta_t * sin_theta_t);

    // Use the Fresnel equations to calculate the reflectance
    float r_parallel = ((refractive_index * cos_theta_i) - cos_theta_t) / ((refractive_index * cos_theta_i) + cos_theta_t);
    float r_perpendicular = (cos_theta_i - (refractive_index * cos_theta_t)) / (cos_theta_i + (refractive_index * cos_theta_t));

    float reflectance = 0.5f * (r_parallel * r_parallel + r_perpendicular * r_perpendicular);

    return reflectance;
}

void calculateCoatScattering(glm::vec3 incident_direction, glm::vec3 surface_normal, float refractive_index) 
{
    float reflectance = fresnel_reflectance(incident_direction, surface_normal, refractive_index);

    thrust::uniform_real_distribution<float> u01(0, 1);
    float rouletteRandomFloat = u01(rng);

    // If the random number is less than the reflectance, reflect the ray; otherwise, absorb it
    if (rouletteRandomFloat < reflectance) {
        return  reflectRay(ray_dir, normal);
        mask = glm::vec3(1, 1, 1);
    }
    else {
        return calculateRandomDirectionInHemisphere(normal, rng);
    }
}
__forceinline__ __host__ __device__

__forceinline__ __host__ __device__
glm::vec3 calculateMetalScattering(glm::vec3 normal, glm::vec3 ray_dir, float phong_exponent, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float phi = TWO_PI * u01(rng);
    float importance_sampled_cosine = u01(rng);
    float cosTheta = powf(importance_sampled_cosine, 1.0f / phong_exponent);
    float sinTheta = sqrtf(1 - cosTheta * cosTheta);

    glm::vec3 w = glm::normalize(reflectRay(ray_dir, normal));
    glm::vec3 u = glm::normalize(glm::cross((ABS(w.x) > .1 ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)), w));
    glm::vec3 v = glm::cross(w, u);

    return u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta;
}
