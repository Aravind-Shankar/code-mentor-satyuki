#pragma once

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <chrono>

#include "Renderer.h"
#include "utility.h"
#include "GPUKernels.cuh"

void Renderer::renderImage()
{
    char bmpHeader[] = {
        'B', 'M',                   // Signature
        0, 0, 0, 0,                  // File size (placeholder)
        0, 0, 0, 0,                  // Reserved
        54, 0, 0, 0,                // Data offset
        40, 0, 0, 0,                // Header size
        static_cast<char>(RESOLUTION_X),   // Image width
        static_cast<char>(RESOLUTION_X >> 8),
        static_cast<char>(RESOLUTION_X >> 16),
        static_cast<char>(RESOLUTION_X >> 24),
        static_cast<char>(RESOLUTION_Y),  // Image height
        static_cast<char>(RESOLUTION_Y >> 8),
        static_cast<char>(RESOLUTION_Y >> 16),
        static_cast<char>(RESOLUTION_Y >> 24),
        1, 0,                       // Number of color planes
        24, 0,                      // Bits per pixel (24 for RGB)
        0, 0, 0, 0,                 // Compression method (0 for no compression)
        0, 0, 0, 0,                 // Image size (placeholder)
        0, 0, 0, 0,                 // Horizontal resolution (pixels per meter)
        0, 0, 0, 0,                 // Vertical resolution (pixels per meter)
        0, 0, 0, 0,                 // Number of colors in the palette
        0, 0, 0, 0                  // Number of important colors
    };

    std::ofstream outFile("Render.bmp", std::ios::binary);

    outFile.write(bmpHeader, sizeof(bmpHeader));

    for (int y = 0; y < RESOLUTION_Y; ++y) {
        for (int x = 0; x < RESOLUTION_X; ++x)
        {
            float div = 1/(float)ITER;
            glm::vec3 color = (render_data.dev_image_data->pool[x + y * RESOLUTION_X].color*div)*255.0f;
            char pixel[] = { color.x,  color.y, color.z};
            outFile.write(pixel, sizeof(pixel));
        }
    }

    int fileSize = 54 + 3 * RESOLUTION_X * RESOLUTION_Y;
    int imageSize = 3 * RESOLUTION_X * RESOLUTION_Y;
    outFile.seekp(2);
    outFile.write(reinterpret_cast<const char*>(&fileSize), 4);
    outFile.seekp(34);
    outFile.write(reinterpret_cast<const char*>(&imageSize), 4);

    outFile.close();
}

void Renderer::allocateOnGPU(Scene &scene)
{
    GPUMemoryPool<Model> gpu_memory_pool0;
    render_data.dev_model_data = gpu_memory_pool0.getInstance();
    render_data.dev_model_data->allocate(scene.models);

    GPUMemoryPool<Mesh> gpu_memory_pool1;
    render_data.dev_mesh_data = gpu_memory_pool1.getInstance();
    render_data.dev_mesh_data->allocate(scene.meshes);

    GPUMemoryPool<Vertex> gpu_memory_pool2;
    render_data.dev_per_vertex_data= gpu_memory_pool2.getInstance();
    render_data.dev_per_vertex_data->allocate(scene.vertices);

    GPUMemoryPool<Triangle> gpu_memory_pool3;
    render_data.dev_triangle_data = gpu_memory_pool3.getInstance();
    render_data.dev_triangle_data->allocate(scene.triangles);

    GPUMemoryPool<Grid> gpu_memory_pool4;
    render_data.dev_grid_data = gpu_memory_pool4.getInstance();
    render_data.dev_grid_data->allocate(scene.grids);

    GPUMemoryPool<Voxel> gpu_memory_pool5;
    render_data.dev_voxel_data = gpu_memory_pool5.getInstance();
    render_data.dev_voxel_data->allocate(scene.voxels);

    GPUMemoryPool<EntityIndex> gpu_memory_pool6;
    render_data.dev_per_voxel_data = gpu_memory_pool6.getInstance();
    render_data.dev_per_voxel_data->allocate(scene.per_voxel_data_pool);

    GPUMemoryPool<Ray> gpu_memory_pool7;
    vector<Ray> rays(RESOLUTION_X * RESOLUTION_Y * SAMPLESX * SAMPLESY);
    render_data.dev_ray_data = gpu_memory_pool7.getInstance();
    render_data.dev_ray_data->allocate(rays);

    GPUMemoryPool<Pixel> gpu_memory_pool8;
    vector<Pixel> image_data(RESOLUTION_X * RESOLUTION_Y);
    render_data.dev_image_data = gpu_memory_pool8.getInstance();
    render_data.dev_image_data->allocate(image_data);

    GPUMemoryPool<int> gpu_memory_pool9;
    vector<int> stencil_data(RESOLUTION_X * RESOLUTION_Y * SAMPLESX * SAMPLESY);
    render_data.dev_stencil = gpu_memory_pool9.getInstance();
    render_data.dev_stencil->allocate(stencil_data);

    GPUMemoryPool<IntersectionData> gpu_memory_pool10;
    vector<IntersectionData> intersection_data(RESOLUTION_X * RESOLUTION_Y*SAMPLESX*SAMPLESY);
    render_data.dev_intersection_data = gpu_memory_pool10.getInstance();
    render_data.dev_intersection_data->allocate(intersection_data);

    GPUMemoryPool<IntersectionData> gpu_memory_pool11;
    render_data.dev_first_intersection_cache = gpu_memory_pool11.getInstance();
    render_data.dev_first_intersection_cache->allocate(intersection_data);

    printCUDAMemoryInfo();
}

void Renderer::free()
{
    render_data.dev_grid_data->free();
    render_data.dev_mesh_data->free();
    render_data.dev_model_data->free();
    render_data.dev_voxel_data->free();
    render_data.dev_per_vertex_data->free();
    render_data.dev_per_voxel_data->free();
    render_data.dev_triangle_data->free();
    render_data.dev_intersection_data->free();
    render_data.dev_first_intersection_cache->free();
    render_data.dev_image_data->free();
    render_data.dev_ray_data->free();
    render_data.dev_stencil->free();
}

void Renderer::renderLoop()
{
    int nrays = render_data.dev_ray_data->size;
    cudaError_t err;

    dim3 threads(32);
    dim3 blocks = (ceil(nrays / 32));

    auto start_time = std::chrono::high_resolution_clock::now();
    long intersection_time = 0;

    initImageKernel << <blocks, threads >> > (nrays, render_data);
    err = cudaDeviceSynchronize();

    bool is_first_intersection_cached = false;

    for (int iter = 0; iter < ITER; iter++)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        bool iterationComplete = false;
        int ibounce = 0;

        nrays = render_data.dev_ray_data->size;
        generateRaysKernel << <blocks, threads >> > (nrays, render_data);
        err = cudaDeviceSynchronize();

        while (!iterationComplete)
        {
            if (ibounce == 0)
            {
                if (is_first_intersection_cached)
                {
                    int size = render_data.dev_first_intersection_cache->size;
                    err = cudaMemcpy(render_data.dev_intersection_data->pool, render_data.dev_first_intersection_cache->pool, sizeof(IntersectionData) * size, cudaMemcpyDeviceToDevice);
                }
                else
                {
                    //cache first intersection;
                    auto start_time = std::chrono::high_resolution_clock::now();
                    computeRaySceneIntersectionKernel << <blocks, threads >> > (nrays, render_data);
                    err = cudaDeviceSynchronize();
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                    intersection_time += duration.count();
                    std::cout << "First Intersection: " << duration.count() << " microseconds" << std::endl;
                    int size = render_data.dev_intersection_data->size;
                    err = cudaMemcpy(render_data.dev_first_intersection_cache->pool, render_data.dev_intersection_data->pool, sizeof(IntersectionData) * size, cudaMemcpyDeviceToDevice);
                    if (err == cudaSuccess)
                    {
                        is_first_intersection_cached = true;
                    }
                }
            }
            else
            {
                auto start_time = std::chrono::high_resolution_clock::now();
                computeRaySceneIntersectionKernel << <blocks, threads >> > (nrays, render_data);
                err = cudaDeviceSynchronize();
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                intersection_time += duration.count();
                std::cout << "Non cached Intersection: " << duration.count() << " microseconds" << std::endl;
            }
            
            shadeRayKernel << <blocks, threads >> > (nrays, iter, render_data);
            err = cudaDeviceSynchronize();

            compactStencilKernel << <blocks, threads >> > (nrays, render_data.dev_ray_data->pool, render_data.dev_stencil->pool);
            err = cudaDeviceSynchronize();

            Ray* itr = thrust::stable_partition(thrust::device, render_data.dev_ray_data->pool, render_data.dev_ray_data->pool + nrays, render_data.dev_stencil->pool, hasTerminated());
            int n = itr - render_data.dev_ray_data->pool;
            nrays = n;

            if (nrays == 0)
            {
                iterationComplete = true;
            }
            ibounce++;
        }
        gatherImageDataKernel << <blocks, threads >> > (render_data);
        
        err = cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Iteration "<<iter+1<<": " << duration.count() << " microseconds" << std::endl;
        std::cout << "Intersection time: " << intersection_time << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Full run: " << duration.count() << " microseconds" << std::endl;
}