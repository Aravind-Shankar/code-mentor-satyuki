#pragma once
#define GLM_FORCE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Scene.h"
#include "Primitive.h"
#include "Config.h"
#include "GPUMemoryPool.h"

using namespace Common;
using namespace Camera;
using namespace SceneElements;
using namespace SpatialAcceleration;

struct RenderData
{
	GPUMemoryPool<Model> *dev_model_data;
	GPUMemoryPool<Mesh> *dev_mesh_data;
	GPUMemoryPool<Vertex> *dev_per_vertex_data;
	GPUMemoryPool<Triangle> *dev_triangle_data;
	GPUMemoryPool<Grid> *dev_grid_data;
	GPUMemoryPool<Voxel> *dev_voxel_data;
	GPUMemoryPool<EntityIndex> *dev_per_voxel_data;
	GPUMemoryPool <Ray> *dev_ray_data;
	GPUMemoryPool<IntersectionData>* dev_intersection_data;
	GPUMemoryPool<IntersectionData>* dev_first_intersection_cache;
	GPUMemoryPool <Pixel>* dev_image_data;
	GPUMemoryPool <int>* dev_stencil;
};

class Renderer
{
public:
	__host__ void allocateOnGPU(Scene &scene);
	__host__ void renderLoop();
	__host__ void renderImage();
	__host__ void free();

	RenderData render_data;
};