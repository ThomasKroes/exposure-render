/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <map>

#include "Exception.h"

namespace ExposureRender
{

class CCudaTimer
{
public:
	CCudaTimer(void);
	virtual ~CCudaTimer(void);

	void	StartTimer(void);
	float	StopTimer(void);
	float	ElapsedTime(void);

private:
	bool			m_Started;
	cudaEvent_t 	m_EventStart;
	cudaEvent_t 	m_EventStop;
};

class CUDA
{
public:
	static void HandleCudaError(const cudaError_t& CudaError)
	{
		if (CudaError != cudaSuccess)
			throw(Exception(Enums::Error, cudaGetErrorString(CudaError)));
	}

	static void ThreadSynchronize()
	{
		CUDA::HandleCudaError(cudaThreadSynchronize());
	}

	template<class T> static void Allocate(T*& pDevicePointer, int Num = 1)
	{
		CUDA::ThreadSynchronize();
		HandleCudaError(cudaMalloc((void**)&pDevicePointer, Num * sizeof(T)));

		CUDA::ThreadSynchronize();
	}

	template<class T> static void AllocatePiched(T*& pDevicePointer, const int Pitch, const int Width, const int Height)
	{
		CUDA::ThreadSynchronize();
		HandleCudaError(cudaMallocPitch((void**)&pDevicePointer, (size_t*)&Pitch, Width * sizeof(T), Height));

		CUDA::ThreadSynchronize();
	}
	
	template<class T> static void MemSet(T*& pDevicePointer, const int Value, int Num = 1)
	{
		CUDA::ThreadSynchronize();
		HandleCudaError(cudaMemset((void*)pDevicePointer, Value, (size_t)(Num * sizeof(T))));

		CUDA::ThreadSynchronize();
	}

	template<class T> static void HostToConstantDevice(T* pHost, char* pDeviceSymbol, int Num = 1)
	{
		CUDA::ThreadSynchronize();
		HandleCudaError(cudaMemcpyToSymbol(pDeviceSymbol, pHost, Num * sizeof(T)));

		CUDA::ThreadSynchronize();
	}

	template<class T> static void MemCopyHostToDeviceSymbol(T* pHost, char* pDeviceSymbol, int Num = 1)
	{
		CUDA::ThreadSynchronize();
		HandleCudaError(cudaMemcpyToSymbol(pDeviceSymbol, pHost, Num * sizeof(T)));

		CUDA::ThreadSynchronize();
	}

	template<class T> static void MemCopyDeviceToDeviceSymbol(T* pDevice, char* pDeviceSymbol, int Num = 1)
	{
		CUDA::ThreadSynchronize();
		HandleCudaError(cudaMemcpyToSymbol(pDeviceSymbol, pDevice, Num * sizeof(T), 0, cudaMemcpyDeviceToDevice));

		CUDA::ThreadSynchronize();
	}

	template<class T> static void MemCopyHostToDevice(T* pHost, T* pDevice, int Num = 1)
	{
		CUDA::ThreadSynchronize();
		HandleCudaError(cudaMemcpy(pDevice, pHost, Num * sizeof(T), cudaMemcpyHostToDevice));

		CUDA::ThreadSynchronize();
	}

	template<class T> static void MemCopyDeviceToHost(T* pDevice, T* pHost, int Num = 1)
	{
		CUDA::ThreadSynchronize();
		HandleCudaError(cudaMemcpy(pHost, pDevice, Num * sizeof(T), cudaMemcpyDeviceToHost));

		CUDA::ThreadSynchronize();
	}

	template<class T> static void MemCopyDeviceToDevice(T* pDeviceSource, T* pDeviceDestination, int Num = 1)
	{
		CUDA::ThreadSynchronize();
		HandleCudaError(cudaMemcpy(pDeviceDestination, pDeviceSource, Num * sizeof(T), cudaMemcpyDeviceToDevice));

		CUDA::ThreadSynchronize();
	}

	static void FreeArray(cudaArray*& pCudaArray)
	{
		CUDA::ThreadSynchronize();
		HandleCudaError(cudaFreeArray(pCudaArray));
		pCudaArray = NULL;

		CUDA::ThreadSynchronize();
	}

	template<class T> static void Free(T*& pBuffer)
	{
		if (pBuffer == NULL)
			return;

		CUDA::ThreadSynchronize();
		
		HandleCudaError(cudaFree(pBuffer));
		pBuffer = NULL;

		CUDA::ThreadSynchronize();
	}

	static void UnbindTexture(textureReference& pTextureReference)
	{
		CUDA::ThreadSynchronize();
		HandleCudaError(cudaUnbindTexture(&pTextureReference));

		CUDA::ThreadSynchronize();
	}

	template<class T> static void BindTexture1D(textureReference& TextureReference, int Num, const T* pBuffer, cudaArray*& pCudaArray, cudaTextureFilterMode TextureFilterMode = cudaFilterModeLinear, cudaTextureAddressMode TextureAddressMode = cudaAddressModeClamp, bool Normalized = true)
	{
		CUDA::ThreadSynchronize();

		const cudaChannelFormatDesc ChannelDescription = cudaCreateChannelDesc<T>();

		TextureReference.normalized		= Normalized;
		TextureReference.filterMode		= TextureFilterMode;
		TextureReference.addressMode[0]	= TextureAddressMode;

		CUDA::FreeArray(pCudaArray);

		HandleCudaError(cudaMallocArray(&pCudaArray, &ChannelDescription, Num, 1));
		HandleCudaError(cudaMemcpyToArray(pCudaArray, 0, 0, pBuffer, Num * sizeof(T), cudaMemcpyHostToDevice));
		HandleCudaError(cudaBindTextureToArray(&TextureReference, pCudaArray, &ChannelDescription));

		CUDA::ThreadSynchronize();
	}

	template<class T> static void BindTexture3D(textureReference& TextureReference, int Extent[3], const T* pBuffer, cudaArray*& pCudaArray, cudaTextureFilterMode TextureFilterMode = cudaFilterModeLinear, cudaTextureAddressMode TextureAddressMode = cudaAddressModeClamp, bool Normalized = true)
	{
		CUDA::ThreadSynchronize();

		const cudaChannelFormatDesc ChannelDescription = cudaCreateChannelDesc<T>();

		const cudaExtent CudaExtent = make_cudaExtent(Extent[0], Extent[1], Extent[2]);

		HandleCudaError(cudaMalloc3DArray(&pCudaArray, &ChannelDescription, CudaExtent));

		cudaMemcpy3DParms CopyParams = {0};

		CopyParams.srcPtr		= make_cudaPitchedPtr((void*)pBuffer, CudaExtent.width * sizeof(unsigned short), CudaExtent.width, CudaExtent.height);
		CopyParams.dstArray		= pCudaArray;
		CopyParams.extent		= CudaExtent;
		CopyParams.kind			= cudaMemcpyHostToDevice;
		
		HandleCudaError(cudaMemcpy3D(&CopyParams));

		TextureReference.normalized		= Normalized;
		TextureReference.filterMode		= TextureFilterMode;      
		TextureReference.addressMode[0]	= TextureAddressMode;  
		TextureReference.addressMode[1]	= TextureAddressMode;
  		TextureReference.addressMode[2]	= TextureAddressMode;

		HandleCudaError(cudaBindTextureToArray(&TextureReference, pCudaArray, &ChannelDescription));

		CUDA::ThreadSynchronize();
	}

	static void GetSymbolAddress(void** pDevicePointer, char* pSymbol)
	{
		CUDA::ThreadSynchronize();
		HandleCudaError(cudaGetSymbolAddress(pDevicePointer, pSymbol));
	}
};

template<class T, int MaxSize = 64>
struct List
{
	int Count;

	HOST List()
	{
		this->Count = 0;
	}

	HOST_DEVICE const T& Get(const int& i)
	{
		return this->Items[i];
	}

	HOST void Add(const T& Resource)
	{
		if (this->Count + 1 >= MaxSize)
			return;

		this->Items[this->Count] = Resource;
		this->Count++;
	}

	HOST void Reset()
	{
		this->Count = 0;
	}

private:
	T Items[MaxSize];
};

template<typename T, int MaxSize>
struct CudaList
{
	typename std::map<int, T>				Resources;
	int										Counter;
	char									DeviceSymbol[MAX_CHAR_SIZE];
	List<T, MaxSize>*						DeviceAllocation;
	List<T, MaxSize>						List;
	typename std::map<int, T>::iterator		It;

	HOST CudaList(const char* pDeviceSymbol)
	{
		this->Counter = 0;

		sprintf_s(DeviceSymbol, MAX_CHAR_SIZE, "%s", pDeviceSymbol);

		this->DeviceAllocation = NULL;
	}

	HOST ~CudaList()
	{
//		CUDA::Free(this->DeviceAllocation);
	}
	
	HOST bool Exists(int ID)
	{
		if (ID < 0)
			return false;

		It = Resources.find(ID);

		return It != Resources.end();
	}

	HOST void Bind(const T& Resource, int& ID)
	{
		if (this->Resources.size() >= MaxSize)
			throw(ErException(Enums::Warning, "Maximum number of resources reached"));

		const bool Exists = this->Exists(ID);

		this->Resources[Counter] = Resource;

		if (!Exists)
		{
			ID = Counter;
			Counter++;
		}

		this->Synchronize();
	}

	HOST void Unbind(int ID)
	{
		if (!this->Exists(ID))
			return;

		It = this->Resources.find(ID);

		if (It != Resources.end())
			Resources.erase(It);

		this->Synchronize();
	}

	HOST void Synchronize()
	{
		if (Resources.empty())
			return;

		this->List.Reset();

		for (It = Resources.begin(); It != Resources.end(); It++)
			this->List.Add(It->second);
		
		if (this->DeviceAllocation == NULL)
			CUDA::Allocate(this->DeviceAllocation);

		CUDA::MemCopyHostToDevice(&this->List, DeviceAllocation);

		void* pSymbol = NULL;

		CUDA::MemCopyDeviceToDeviceSymbol(&DeviceAllocation, pSymbol);
	}

	HOST T& operator[](const int& i)
	{
		It = this->Resources.find(i);

		if (It == Resources.end())
			throw(ErException(Enums::Fatal, "Resource does not exist"));

		return this->Resources[i];
	}
};

#define LAUNCH_CUDA_KERNEL_TIMED(cudakernelcall, title)									\
{																						\
	cudaEvent_t EventStart, EventStop;													\
																						\
	CUDA::HandleCudaError(cudaEventCreate(&EventStart));								\
	CUDA::HandleCudaError(cudaEventCreate(&EventStop));									\
	CUDA::HandleCudaError(cudaEventRecord(EventStart, 0));								\
																						\
	cudakernelcall;																		\
																						\
	CUDA::HandleCudaError(cudaGetLastError());											\
	CUDA::HandleCudaError(cudaThreadSynchronize());										\
																						\
	CUDA::HandleCudaError(cudaEventRecord(EventStop, 0));								\
	CUDA::HandleCudaError(cudaEventSynchronize(EventStop));								\
																						\
	float TimeDelta = 0.0f;																\
																						\
	CUDA::HandleCudaError(cudaEventElapsedTime(&TimeDelta, EventStart, EventStop));		\
																						\
	/*gKernelTimings.Add(ErKernelTiming(title, TimeDelta));*/							\
																						\
	CUDA::HandleCudaError(cudaEventDestroy(EventStart));								\
	CUDA::HandleCudaError(cudaEventDestroy(EventStop));									\
}

#define LAUNCH_CUDA_KERNEL(cudakernelcall)												\
{																						\
	cudakernelcall;																		\
																						\
	CUDA::HandleCudaError(cudaGetLastError());											\
	CUDA::HandleCudaError(cudaThreadSynchronize());										\
}

}