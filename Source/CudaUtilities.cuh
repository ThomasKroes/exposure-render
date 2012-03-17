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

bool InitializeCuda(void);
int GetTotalCudaMemory(void);
int GetAvailableCudaMemory(void);
int GetUsedCudaMemory(void);
int GetMaxGigaFlopsDeviceID(void);
bool SetCudaDevice(const int& CudaDeviceID);
void ResetDevice(void);

/*
struct CudaErrorString
{
	cudaError_t		ID;
	char			Description[MAX_CHAR_SIZE]
};

ErrorString CudaErrorStrings[] =
{
	{ 1, "The device function being invoked (usually via ::cudaLaunch()) was not previously configured via the ::cudaConfigureCall() function." },
	{ 2, "The API call failed because it was unable to allocate enough memory to perform the requested operation." },
	{ 3, "The API call failed because the CUDA driver and runtime could not be initialized." },
	{ 4, "An exception occurred on the device while executing a kernel." },
	{ 5, "This indicated that a previous kernel launch failed. " },
	{ 6, "This indicates that the device kernel took too long to execute." },
	{ 7, "This indicates that a launch did not occur because it did not have appropriate resources." },
	{ 8, "The requested device function does not exist or is not compiled for the proper device architecture." },
	{ 9, "This indicates that a kernel launch is requesting resources that can never be satisfied by the current device." },
	{ 10, "This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device." },
	{ 11, "This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values." },
	{ 12, "This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable range for pitch." },
	{ 13, "This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier." },
	{ 14, "This indicates that the buffer object could not be mapped." },
	{ 15, "This indicates that the buffer object could not be unmapped." },
	{ 16, "This indicates that at least one host pointer passed to the API call is not a valid host pointer." },
	{ 17, "This indicates that at least one device pointer passed to the API call is not a valid device pointer." },
	{ 18, "This indicates that the texture passed to the API call is not a valid texture." },
	{ 19, "This indicates that the texture binding is not valid." },
	{ 20, "This indicates that the channel descriptor passed to the API call is not valid." },
	{ 21, "This indicates that the direction of the memcpy passed to the API call is not one of the types specified by ::cudaMemcpyKind." },
	{ 22, "This indicated that the user has taken the address of a constant variable, which was forbidden up until the CUDA 3.1 release." },
	{ 23, "This indicated that a texture fetch was not able to be performed." },
	{ 24, "This indicated that a texture was not bound for access." },
	{ 25, "This indicated that a synchronization operation had failed." },
	{ 26, "This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA." },
	{ 27, "This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA." },
	{ 28, "Mixing of device and device emulation code was not allowed." },
	{ 29, "This indicates that a CUDA Runtime API call cannot be executed because it is being called during process shut down, at a point in time after CUDA driver has been unloaded." },
	{ 30, "This indicates that an unknown internal error has occurred." },
	{ 31, "This indicates that the API call is not yet implemented." },
	{ 32, "This indicated that an emulated device pointer exceeded the 32-bit address range." },
	{ 33, "This indicates that a resource handle passed to the API call was not valid." },
	{ 34, "This indicates that asynchronous operations issued previously have not completed yet." },
	{ 35, "This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library." },
	{ 36, "This indicates that the user has called ::cudaSetValidDevices(), ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(), ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by calling non-device management operations (allocating memory and launching kernels are examples of non-device management operations)." },
	{ 37, "This indicates that the surface passed to the API call is not a valid surface." },
	{ 38, "This indicates that no CUDA-capable devices were detected by the installed CUDA driver." },
	{ 39, "This indicates that an uncorrectable ECC error was detected during execution." },
	{ 40, "This indicates that a link to a shared object failed to resolve." },
	{ 41, "This indicates that initialization of a shared object failed." },
	{ 42, "This indicates that the ::cudaLimit passed to the API call is not supported by the active device." },
	{ 43, "This indicates that multiple global or constant variables (across separate CUDA source files in the application) share the same string name." },
	{ 44, "This indicates that multiple textures (across separate CUDA source files in the application) share the same string name." },
	{ 45, "This indicates that multiple surfaces (across separate CUDA source files in the application) share the same string name." },
	{ 46, "This indicates that all CUDA devices are busy or unavailable at the current time." },
	{ 47, "This indicates that the device kernel image is invalid." },
	{ 48, "This indicates that there is no kernel image available that is suitable for the device." },
	{ 49, "This indicates that the current context is not compatible with this the CUDA Runtime." },
	{ 50, "This error indicates that a call to ::cudaDeviceEnablePeerAccess() is trying to re-enable peer addressing on from a context which has already had peer addressing enabled." },
	{ 51, "This error indicates that ::cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been enabled yet via ::cudaDeviceEnablePeerAccess()." },
	{ 54, "This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread." },
	{ 55, "This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread." },
	{ 56, "This indicates profiler has not been initialized yet." },
	{ 57, "This indicates profiler is already started." },
	{ 58, "This indicates profiler is already stopped." },
	{ 59, "An assert triggered in device code during kernel execution." },
	{ 60, "This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to ::cudaEnablePeerAccess()." },
	{ 61, "This error indicates that the memory range passed to ::cudaHostRegister() has already been registered." },
	{ 62, "This error indicates that the pointer passed to ::cudaHostUnregister() does not correspond to any currently registered memory region." },
	{ 63, "This error indicates that an OS call failed." },
	{ 64, "This error indicates that an OS call failed." },
	{ 127, "This indicates an internal startup failure in the CUDA runtime." },
	{ 10000, "Any unhandled CUDA driver error is added to this value and returned via the runtime." },
	{ 0, "" }
};

void GetCudaErrorString(const cudaError_t& CudaError, char* pErrorString)
{
	
	while ()
}
*/

class CUDA
{
public:
	static void HandleCudaError(const cudaError_t& CudaError)
	{
		if (CudaError != cudaSuccess)
			throw("CUDA Error", cudaGetErrorString(CudaError));
	}

	template<class T> static void HostToConstantDevice(T* pHost, char* pSymbol, int Num = 1)
	{
		HandleCudaError(cudaMemcpyToSymbol(pSymbol, pHost, Num * sizeof(T)));
	}

	template<class T> static void MemCopyHostToDevice(T* pHost, T* pDevice, int Num = 1)
	{
		HandleCudaError(cudaMemcpy(pDevice, pHost, Num * sizeof(T), cudaMemcpyHostToDevice));
	}

	template<class T> static void MemCopyDeviceToHost(T* pDevice, T* pHost, int Num = 1)
	{
		HandleCudaError(cudaMemcpy(pHost, pDevice, Num * sizeof(T), cudaMemcpyDeviceToHost));
	}

	template<class T> static void MemCopyDeviceToDevice(T* pDeviceSource, T* pDeviceDestination, int Num = 1)
	{
		HandleCudaError(cudaMemcpy(pDeviceDestination, pDeviceSource, Num * sizeof(T), cudaMemcpyDeviceToDevice));
	}

	static void FreeArray(cudaArray*& pCudaArray)
	{
		HandleCudaError(cudaFreeArray(pCudaArray));
		pCudaArray = NULL;
	}

	static void UnbindTexture(textureReference& pTextureReference)
	{
		HandleCudaError(cudaUnbindTexture(&pTextureReference));
	}


	template<class T> static void BindTexture3D(textureReference& TextureReference, int Extent[3], const T* pBuffer, cudaArray*& pCudaArray, cudaTextureFilterMode TextureFilterMode = cudaFilterModeLinear, cudaTextureAddressMode TextureAddressMode = cudaAddressModeClamp, bool Normalized = true)
	{
		cudaChannelFormatDesc ChannelDescription = cudaCreateChannelDesc<T>();

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
	}
};
