/*
    Exposure Render: An interactive photo-realistic volume rendering framework
    Copyright (C) 2011 Thomas Kroes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "exception.h"
#include "log.h"

#ifdef __CUDA_ARCH__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <map>

using namespace std;

namespace ExposureRender
{

namespace Cuda
{

static inline void HandleCudaError(const cudaError_t& CudaError, const char* pTitle = "")
{
	char Message[256];

	snprintf(Message, 256, "%s (%s)", cudaGetErrorString(CudaError), pTitle);

	if (CudaError != cudaSuccess)
		throw(Exception(Enums::Error, Message));
}

static inline void ThreadSynchronize()
{
	Cuda::HandleCudaError(cudaThreadSynchronize(), "cudaThreadSynchronize");
}

template<class T> static inline void Allocate(T*& pDevicePointer, int Num = 1)
{
	Cuda::ThreadSynchronize();
	HandleCudaError(cudaMalloc((void**)&pDevicePointer, Num * sizeof(T)), "cudaMalloc");
	Cuda::ThreadSynchronize();
}

template<class T> static inline void AllocatePiched(T*& pDevicePointer, const int Pitch, const int Width, const int Height)
{
	Cuda::ThreadSynchronize();
	HandleCudaError(cudaMallocPitch((void**)&pDevicePointer, (size_t*)&Pitch, Width * sizeof(T), Height), "cudaMallocPitch");
	Cuda::ThreadSynchronize();
}

template<class T> static inline void MemSet(T*& pDevicePointer, const int Value, int Num = 1)
{
	Cuda::ThreadSynchronize();
	HandleCudaError(cudaMemset((void*)pDevicePointer, Value, (size_t)(Num * sizeof(T))), "cudaMemset");
	Cuda::ThreadSynchronize();
}

template<class T> static inline void HostToConstantDevice(T* pHost, char* pDeviceSymbol, int Num = 1)
{
	Cuda::ThreadSynchronize();
	HandleCudaError(cudaMemcpyToSymbol(pDeviceSymbol, pHost, Num * sizeof(T)), "cudaMemcpyToSymbol");
	Cuda::ThreadSynchronize();
}

template<class T> static inline void MemCopyHostToDeviceSymbol(T* pHost, const char* pDeviceSymbol, const int& Num = 1, const int& Offset = 0)
{
	Cuda::ThreadSynchronize();
	HandleCudaError(cudaMemcpyToSymbol(pDeviceSymbol, pHost, Num * sizeof(T), Offset, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");
	Cuda::ThreadSynchronize();
}

template<class T> static inline void MemCopyDeviceToDeviceSymbol(T* pDevice, const char* pDeviceSymbol, const int& Num = 1, const int& Offset = 0)
{
	Cuda::ThreadSynchronize();
	HandleCudaError(cudaMemcpyToSymbol(pDeviceSymbol, pDevice, Num * sizeof(T), 0, cudaMemcpyDeviceToDevice), "cudaMemcpyToSymbol");
	Cuda::ThreadSynchronize();
}

template<class T> static inline void MemCopyHostToDevice(T* pHost, T* pDevice, int Num = 1)
{
	Cuda::ThreadSynchronize();
	HandleCudaError(cudaMemcpy(pDevice, pHost, Num * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
	Cuda::ThreadSynchronize();
}

template<class T> static inline void MemCopyDeviceToHost(T* pDevice, T* pHost, int Num = 1)
{
	Cuda::ThreadSynchronize();
	HandleCudaError(cudaMemcpy(pHost, pDevice, Num * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
	Cuda::ThreadSynchronize();
}

template<class T> static inline void MemCopyDeviceToDevice(T* pDeviceSource, T* pDeviceDestination, int Num = 1)
{
	Cuda::ThreadSynchronize();
	HandleCudaError(cudaMemcpy(pDeviceDestination, pDeviceSource, Num * sizeof(T), cudaMemcpyDeviceToDevice), "cudaMemcpy");
	Cuda::ThreadSynchronize();
}

static inline void FreeArray(cudaArray*& pCudaArray)
{
	Cuda::ThreadSynchronize();
	HandleCudaError(cudaFreeArray(pCudaArray), "cudaFreeArray");
	pCudaArray = NULL;
	Cuda::ThreadSynchronize();
}

template<class T> static inline void Free(T*& pBuffer)
{
	if (pBuffer == NULL)
		return;

	Cuda::ThreadSynchronize();
	
	HandleCudaError(cudaFree(pBuffer), "cudaFree");
	pBuffer = NULL;

	Cuda::ThreadSynchronize();
}

static inline void GetSymbolAddress(void** pDevicePointer, char* pSymbol)
{
	Cuda::ThreadSynchronize();
	HandleCudaError(cudaGetSymbolAddress(pDevicePointer, pSymbol), "cudaGetSymbolAddress");
}

}

}

#endif