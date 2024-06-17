#pragma once
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

using namespace std;

template <typename T>
class GPUMemoryPool
{
public:
	GPUMemoryPool() :size(0), pool(nullptr),instance(nullptr) {};
	GPUMemoryPool* getInstance()
	{
		if (instance == nullptr)
		{
			cudaError_t err = cudaMallocManaged(&instance, sizeof(GPUMemoryPool<T>));
		}
		return instance;
	}
	void allocate(const vector<T>& data)
	{
		//if (instance)
		{
			size = data.size();
			cudaError_t err = cudaMallocManaged(&pool, sizeof(T)*size);

			const T* pData = data.data();
			err = cudaMemcpy(pool, pData, sizeof(T) * size, cudaMemcpyHostToDevice);
		}
	}
	void free()
	{
		cudaFree(pool);
		if (instance) cudaFree(instance);
	}

	int size;
	T* pool;

private:
	//GPUMemoryPool() :size(0), pool(nullptr) {};
	GPUMemoryPool* instance;
};

//template <typename T>
//GPUMemoryPool<T>* GPUMemoryPool<T>::instance = nullptr;