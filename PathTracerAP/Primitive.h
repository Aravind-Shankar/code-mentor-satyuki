#pragma once
#include <iostream>
#include <vector>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Config.h"

namespace Common
{
	typedef int EntityIndex;

	struct IndexRange
	{
		int start_index;
		int end_index;
	};
}

namespace Geometry
{
	struct Vertex
	{
		glm::vec3 position;
		glm::vec3 normal;
		glm::vec2 uv;
	};

	struct Triangle
	{
		int vertex_indices[3];
	};

	struct BoundingBox
	{
		glm::vec3 min, max;
		BoundingBox()
		{
			//Initialize bounding box
			min.x = FLOAT_MAX;
			min.y = FLOAT_MAX;
			min.z = FLOAT_MAX;

			max.x = FLOAT_MIN;
			max.y = FLOAT_MIN;
			max.z = FLOAT_MIN;
		}

		void update(glm::vec3 vertex)
		{
			min.x = min.x > vertex.x ? vertex.x : min.x;
			min.y = min.y > vertex.y ? vertex.y : min.y;
			min.z = min.z > vertex.z ? vertex.z : min.z;

			max.x = max.x < vertex.x ? vertex.x : max.x;
			max.y = max.y < vertex.y ? vertex.y : max.y;
			max.z = max.z < vertex.z ? vertex.z : max.z;
		}
	};
}

namespace SceneElements
{
	using namespace Common;
	using namespace Geometry;

	struct Material
	{
		enum MaterialType 
		{
			DIFFUSE,
			SPECULAR,
			REFLECTIVE,
			REFRACTIVE,
			EMISSIVE,
			COAT,
			METAL
		} material_type;

		float refractive_index;
		float phong_exponent;
		glm::vec3 color;
	};

	struct Mesh
	{
		IndexRange vertex_indices;
		IndexRange triangle_indices;
		BoundingBox bounding_box;
	};


	struct Model
	{
		int grid_index;
		int mesh_index;
		glm::mat4 model_to_world;
		glm::mat4 world_to_model;
		Material mat;
	};

}

namespace SpatialAcceleration
{
	using namespace Common;

	enum EntityType {
		MODEL,
		SCENE,
		TRIANGLE,
		SPHERE
	};

	struct Voxel3DIndex
	{
		int x;
		int y;
		int z;
	};

	struct Voxel
	{
		IndexRange entity_index_range;
		EntityType entity_type;
	};

	struct Grid
	{
		IndexRange voxelIndices;
		struct VoxelWidth
		{
			float x, y, z;
		}voxel_width;

		EntityType entity_type;
		int entity_index;
	};
}

namespace Camera
{
	using namespace SceneElements;
	struct Pixel
	{
		glm::vec3 color;
	};

	struct IntersectionData
	{
		float impact_distance;
		glm::vec3 impact_normal;
		Material impact_mat;
		int ipixel;
	};

	struct Ray
	{
		struct Points
		{
			glm::vec3 orig;
			glm::vec3 dir;
		} base, transformed;

		struct Cache
		{
			glm::vec3 inv_dir;
		}cache;

		struct MetaData
		{
			int ipixel;
			int remaining_bounces;
		}meta_data;

		glm::vec3 color;
	};
}
