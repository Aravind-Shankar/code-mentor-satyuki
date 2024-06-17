#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>      
#include <assimp/scene.h> 
#include <assimp/postprocess.h> 

#include "Primitive.h"

using namespace std;
using namespace Common;
using namespace Geometry;
using namespace SceneElements;
using namespace SpatialAcceleration;

class Scene
{
public:
	Scene(string config);

	vector<Model> models;
	vector<Mesh> meshes;
	vector<Vertex> vertices;
	vector<Triangle> triangles;
	vector<Grid> grids;
	vector<Voxel> voxels;
	vector<EntityIndex> per_voxel_data_pool;

private:
	void loadAndProcessMeshFile(string path, Mesh& mesh);
	void addMeshesToGrid();
	void processMesh(aiMesh* ai_mesh, Mesh& mesh, const aiScene* scene);
	void processNode(aiNode* node, Mesh& mesh, const aiScene* scene);
};

