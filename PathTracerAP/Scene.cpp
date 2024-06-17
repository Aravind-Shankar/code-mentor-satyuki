#include "Scene.h"
#define CLAMP(value, min, max) ((value) < (min) ? (min) : ((value) > (max) ? (max) : (value)))
Scene::Scene(string config)
{

    Mesh box;
    loadAndProcessMeshFile("Input data\\enclosing_box.obj", box);
    meshes.push_back(box);

    Mesh ceiling_light;
    loadAndProcessMeshFile("Input data\\ceiling_light.obj", ceiling_light);
    meshes.push_back(ceiling_light);

    Mesh bunny_mesh;
    loadAndProcessMeshFile("Input data\\stanford_bunny.obj", bunny_mesh);
    meshes.push_back(bunny_mesh);

    Mesh armadillo_mesh;
    loadAndProcessMeshFile("Input data\\stanford_armadillo.obj", armadillo_mesh);
    meshes.push_back(armadillo_mesh);

    Mesh dragon_mesh;
    loadAndProcessMeshFile("Input data\\stanford_dinosaur.obj", dragon_mesh);
    meshes.push_back(dragon_mesh);

    glm::mat4 scale_matrix, rotate_matrix, translation_matrix;

    Model bunny_model_1;
    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.1f, 0.1f, 0.1f));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(60.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(75.0f, -75.0f, 0.0f));
    bunny_model_1.mesh_index = 2;
    bunny_model_1.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    bunny_model_1.world_to_model = glm::inverse(bunny_model_1.model_to_world);
    bunny_model_1.mat.color = glm::vec3(0.99f, 0.99f, 0.001f);
    bunny_model_1.mat.material_type = Material::MaterialType::COAT;
    bunny_model_1.mat.phong_exponent = 80.0f;
    models.push_back(bunny_model_1);

    Model bunny_model_2;
    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.1f, 0.1f, 0.1f));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(325.0f, -35.0f, 0.0f));
    bunny_model_2.mesh_index = 2;
    bunny_model_2.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    bunny_model_2.world_to_model = glm::inverse(bunny_model_2.model_to_world);
    bunny_model_2.mat.color = glm::vec3(0.99f, 0.99f, 0.75f);
    bunny_model_2.mat.material_type = Material::MaterialType::COAT;
    bunny_model_2.mat.phong_exponent = 80.0f;
    models.push_back(bunny_model_2);

    Model armadillo_model;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.1f, 0.1f, 0.1f));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(225.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(-100.0f, 5.0f, 0.0f));
    armadillo_model.mesh_index = 3;
    armadillo_model.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    armadillo_model.world_to_model = glm::inverse(armadillo_model.model_to_world);
    armadillo_model.mat.color = glm::vec3(0.99f, 0.05f, 0.99f);
    armadillo_model.mat.material_type = Material::MaterialType::COAT;
    armadillo_model.mat.phong_exponent = 20.0f;
    models.push_back(armadillo_model);

    Model box_model;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.1, 0.1, 0.1));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(25.0f, -120.0f, 0.0f));
    box_model.mesh_index = 0;
    box_model.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    box_model.world_to_model = glm::inverse(box_model.model_to_world);
    box_model.mat.color = glm::vec3(0.99f, 0.99f, 0.99f);
    box_model.mat.material_type = Material::MaterialType::COAT;
    box_model.mat.phong_exponent = 10.0f;
    models.push_back(box_model);

    Model dragon_model;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.1, 0.1, 0.1));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(245.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(-300.0f, 25.0f, 50.0f));
    dragon_model.mesh_index = 4;
    dragon_model.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    dragon_model.world_to_model = glm::inverse(dragon_model.model_to_world);
    dragon_model.mat.color = glm::vec3(0.001f, 0.5f, 0.99f);
    dragon_model.mat.material_type = Material::MaterialType::COAT;
    dragon_model.mat.phong_exponent = 50.0f;
    models.push_back(dragon_model);

    Model stand_model;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.1, 0.1, 0.1));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(325.0f, -90.0f, 0.0f));
    stand_model.mesh_index = 1;
    stand_model.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    stand_model.world_to_model = glm::inverse(stand_model.model_to_world);
    stand_model.mat.color = glm::vec3(0.99f, 0.50f, 0.60f);
    stand_model.mat.material_type = Material::MaterialType::COAT;
    stand_model.mat.phong_exponent = 50.0f;
    models.push_back(stand_model);

    Model stand_model2;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.1, 0.1, 0.1));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(30.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(75.0f, -120.0f, 0.0f));
    stand_model2.mesh_index = 1;
    stand_model2.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    stand_model2.world_to_model = glm::inverse(stand_model2.model_to_world);
    stand_model2.mat.color = glm::vec3(0.99f, 0.05f, 0.10f);
    stand_model2.mat.material_type = Material::MaterialType::COAT;
    stand_model2.mat.phong_exponent = 50.0f;
    models.push_back(stand_model2);

    Model light_model;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.2, 0.1, 0.2));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 850.0f, -100.0f));
    light_model.mesh_index = 1;
    light_model.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    light_model.world_to_model = glm::inverse(light_model.model_to_world);
    light_model.mat.color = glm::vec3(0.99f, 0.99f, 0.99f);
    light_model.mat.material_type = Material::MaterialType::EMISSIVE;
    models.push_back(light_model);

    Model light_model2;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.2, 0.2, 0.1));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 375.0f, 950.0f));
    light_model2.mesh_index = 1;
    light_model2.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    light_model2.world_to_model = glm::inverse(light_model2.model_to_world);
    light_model2.mat.color = glm::vec3(0.99f, 0.99f, 0.99f);
    light_model2.mat.material_type = Material::MaterialType::EMISSIVE;
    models.push_back(light_model2);

    Model light_model3;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.1, 0.2, 0.2));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(-520.0f, 375.0f, 0.0f));
    light_model3.mesh_index = 1;
    light_model3.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    light_model3.world_to_model = glm::inverse(light_model3.model_to_world);
    light_model3.mat.color = glm::vec3(0.99f, 0.99f, 0.99f);
    light_model3.mat.material_type = Material::MaterialType::EMISSIVE;
    models.push_back(light_model3);

    Model light_model4;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.1, 0.2, 0.2));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(550.0f, 375.0f, 0.0f));
    light_model4.mesh_index = 1;
    light_model4.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    light_model4.world_to_model = glm::inverse(light_model4.model_to_world);
    light_model4.mat.color = glm::vec3(0.99f, 0.99f, 0.99f);
    light_model4.mat.material_type = Material::MaterialType::EMISSIVE;
    models.push_back(light_model4);

    addMeshesToGrid();
}

void Scene::loadAndProcessMeshFile(string path, Mesh& mesh)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        cerr << "Error loading mesh: " << importer.GetErrorString() << endl;
        return;
    }

    processNode(scene->mRootNode, mesh, scene);
}

void Scene::processNode(aiNode* node, Mesh& mesh, const aiScene* scene)
{
    for (unsigned int i = 0; i < node->mNumMeshes; ++i)
    {
        aiMesh* ai_mesh = scene->mMeshes[node->mMeshes[i]];
        processMesh(ai_mesh, mesh, scene);
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i)
    {
        processNode(node->mChildren[i], mesh, scene);
    }
}


glm::vec3 convertFromVector3D(const aiVector3D &vector3D)
{
    glm::vec3 vec3;
    vec3.x = vector3D.x * BASE_MODEL_SCALE;
    vec3.y = vector3D.y * BASE_MODEL_SCALE;
    vec3.z = vector3D.z * BASE_MODEL_SCALE;
    return vec3;
}

void Scene::processMesh(aiMesh* ai_mesh, Mesh& mesh, const aiScene* scene)
{
    mesh.vertex_indices.start_index = vertices.size();
    for (unsigned int i = 0; i < ai_mesh->mNumVertices; ++i)
    {
        Vertex vertex_data_obj;
        vertex_data_obj.position = convertFromVector3D(ai_mesh->mVertices[i]);
        vertex_data_obj.normal = convertFromVector3D(ai_mesh->mNormals[i]);
        vertices.push_back(vertex_data_obj);
        mesh.bounding_box.update(vertex_data_obj.position);
    }
    mesh.vertex_indices.end_index = vertices.size();

    mesh.triangle_indices.start_index = triangles.size();
    for (unsigned int i = 0; i < ai_mesh->mNumFaces; ++i)
    {
        aiFace face = ai_mesh->mFaces[i];
        assert(face.mNumIndices == 3);

        Triangle tri;
        for (unsigned int j = 0; j < 3; ++j)
        {
            tri.vertex_indices[j] = mesh.vertex_indices.start_index + face.mIndices[j];
        }
        triangles.push_back(tri);
    }
    mesh.triangle_indices.end_index = triangles.size();
}

void computeVoxelIndex( Voxel3DIndex& min, Voxel3DIndex& max, BoundingBox& bounding_box, const Grid::VoxelWidth &voxel_width, const glm::vec3* triangle)
{
    BoundingBox t_box;
    t_box.update(triangle[0]);
    t_box.update(triangle[1]);
    t_box.update(triangle[2]);

    min.x = floor(abs(bounding_box.min.x - t_box.min.x) / voxel_width.x);
    max.x = floor(abs(bounding_box.min.x - t_box.max.x) / voxel_width.x);

    min.y = floor(abs(bounding_box.min.y - t_box.min.y) / voxel_width.y);
    max.y = floor(abs(bounding_box.min.y - t_box.max.y) / voxel_width.y);

    min.z = floor(abs(bounding_box.min.z - t_box.min.z) / voxel_width.z);
    max.z = floor(abs(bounding_box.min.z - t_box.max.z) / voxel_width.z);

    min.x = CLAMP(min.x, 0, GRID_X - 1);
    min.y = CLAMP(min.y, 0, GRID_Y - 1);
    min.z = CLAMP(min.z, 0, GRID_Z - 1);

    max.x = CLAMP(max.x, 0, GRID_X - 1);
    max.y = CLAMP(max.y, 0, GRID_Y - 1);
    max.z = CLAMP(max.z, 0, GRID_Z - 1);
}

void Scene::addMeshesToGrid()
{
    vector<bool> is_mesh_processed(meshes.size(), false);
    vector<int> grid_index_cache(meshes.size());

    for (int i = 0; i < models.size(); i++)
    {
        if (is_mesh_processed[models[i].mesh_index])
        {
            models[i].grid_index = grid_index_cache[models[i].mesh_index];
            continue;
        }

        is_mesh_processed[models[i].mesh_index] = true;
        grid_index_cache[models[i].mesh_index] = grids.size();
        models[i].grid_index = grids.size();

        Grid grid;
        int mesh_index = models[i].mesh_index;
        grid.entity_type = EntityType::MODEL;
        grid.entity_index = i;
        
        vector<vector<int>> voxels_buffer(GRID_X * GRID_Y * GRID_Z);
        float x_width = meshes[mesh_index].bounding_box.max.x - meshes[mesh_index].bounding_box.min.x;
        float y_width = meshes[mesh_index].bounding_box.max.y - meshes[mesh_index].bounding_box.min.y;
        float z_width = meshes[mesh_index].bounding_box.max.z - meshes[mesh_index].bounding_box.min.z;

        grid.voxel_width.x = x_width / GRID_X;
        grid.voxel_width.y = y_width / GRID_Y;
        grid.voxel_width.z = z_width / GRID_Z;

        for (int t = meshes[mesh_index].triangle_indices.start_index; t < meshes[mesh_index].triangle_indices.end_index; t++)
        {
            int t0_index = triangles[t].vertex_indices[0];
            int t1_index = triangles[t].vertex_indices[1];
            int t2_index = triangles[t].vertex_indices[2];

            glm::vec3 triangle[3];
            triangle[0] = vertices[t0_index].position;
            triangle[1] = vertices[t1_index].position;
            triangle[2] = vertices[t2_index].position;

            Voxel3DIndex min, max;

            computeVoxelIndex(min, max, meshes[mesh_index].bounding_box, grid.voxel_width, triangle);

            for (int z = min.z; z <= max.z; z++)
            {
                for (int y = min.y; y <= max.y; y++)
                {
                    for (int x = min.x; x <= max.x; x++)
                    {
                        int index = x + y * GRID_X + GRID_X * GRID_Y * z;
                        voxels_buffer[index].push_back(t);
                    }
                }
            }
        }

        grid.voxelIndices.start_index = voxels.size();
        for (int i = 0; i < voxels_buffer.size(); i++)
        {
            Voxel voxel;
            voxel.entity_type = EntityType::TRIANGLE;
            voxel.entity_index_range.start_index = per_voxel_data_pool.size();
            for (int j = 0; j < voxels_buffer[i].size(); j++)
            {
                EntityIndex iTriangle;
                iTriangle = voxels_buffer[i][j];
                per_voxel_data_pool.push_back(iTriangle);
            }       
            voxel.entity_index_range.end_index = per_voxel_data_pool.size();
            voxels.push_back(voxel);
        }
        grid.voxelIndices.end_index = voxels.size();

        grids.push_back(grid);
    }
}