/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
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
void HandleCudaError(const cudaError_t CudaError, const char* pDescription = "");
void HandleCudaKernelError(const cudaError_t CudaError, const char* pName = "");
int GetTotalCudaMemory(void);
int GetAvailableCudaMemory(void);
int GetUsedCudaMemory(void);
int GetMaxGigaFlopsDeviceID(void);
bool SetCudaDevice(const int& CudaDeviceID);
void ResetDevice(void);

/*
bool InitializeCuda(void)
{
	// No CUDA enabled devices
	int NoDevices = 0;

	cudaError_t ErrorID = cudaGetDeviceCount(&NoDevices);


	gStatus.SetStatisticChanged("Graphics Card", "No. CUDA capable devices", QString::number(NoDevices));

	Log("Found " + QString::number(NoDevices) + " CUDA enabled device(s)", "graphic-card");

	int DriverVersion = 0, RuntimeVersion = 0; 

	cudaDriverGetVersion(&DriverVersion);
	cudaRuntimeGetVersion(&RuntimeVersion);

	QString DriverVersionString		= QString::number(DriverVersion / 1000) + "." + QString::number(DriverVersion % 100);
	QString RuntimeVersionString	= QString::number(RuntimeVersion / 1000) + "." + QString::number(RuntimeVersion % 100);

	gStatus.SetStatisticChanged("Graphics Card", "CUDA Driver Version", DriverVersionString);
	gStatus.SetStatisticChanged("Graphics Card", "CUDA Runtime Version", RuntimeVersionString);

	Log("Current CUDA driver version: " + DriverVersionString, "graphic-card");
	Log("Current CUDA runtime version: " + RuntimeVersionString, "graphic-card");

	for (int Device = 0; Device < NoDevices; Device++)
	{
		QString DeviceString = "Device " + QString::number(Device);

		gStatus.SetStatisticChanged("Graphics Card", DeviceString, "");

		cudaDeviceProp DeviceProperties;
		cudaGetDeviceProperties(&DeviceProperties, Device);

		QString CudaCapabilityString = QString::number(DeviceProperties.major) + "." + QString::number(DeviceProperties.minor);
		
		gStatus.SetStatisticChanged(DeviceString, "CUDA Capability", CudaCapabilityString);

		// Memory
		gStatus.SetStatisticChanged(DeviceString, "On Board Memory", "", "", "memory");

		gStatus.SetStatisticChanged("On Board Memory", "Total Global Memory", QString::number((float)DeviceProperties.totalGlobalMem / powf(1024.0f, 2.0f), 'f', 2), "MB");
		gStatus.SetStatisticChanged("On Board Memory", "Total Constant Memory", QString::number((float)DeviceProperties.totalConstMem / powf(1024.0f, 2.0f), 'f', 2), "MB");

		int MemoryClock, MemoryBusWidth, L2CacheSize;
		GetCudaAttribute<int>(&MemoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, Device);
		GetCudaAttribute<int>(&MemoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, Device);
		GetCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, Device);

		gStatus.SetStatisticChanged("On Board Memory", "Memory Clock Rate", QString::number(MemoryClock * 1e-3f), "Mhz");
		gStatus.SetStatisticChanged("On Board Memory", "Memory Bus Width", QString::number(MemoryBusWidth), "bit");
		gStatus.SetStatisticChanged("On Board Memory", "L2 Cache Size", QString::number(L2CacheSize), "bytes");
		gStatus.SetStatisticChanged("On Board Memory", "Maximum Memory Pitch", QString::number((float)DeviceProperties.memPitch / powf(1024.0f, 2.0f), 'f', 2), "MB");
		
		// Processor
		gStatus.SetStatisticChanged(DeviceString, "Processor", "", "", "processor");
		gStatus.SetStatisticChanged("Processor", "No. Multiprocessors", QString::number(DeviceProperties.multiProcessorCount), "Processors");
		gStatus.SetStatisticChanged("Processor", "GPU Clock Speed", QString::number(DeviceProperties.clockRate * 1e-6f, 'f', 2), "GHz");
		gStatus.SetStatisticChanged("Processor", "Max. Block Size", QString::number(DeviceProperties.maxThreadsDim[0]) + " x " + QString::number(DeviceProperties.maxThreadsDim[1]) + " x " + QString::number(DeviceProperties.maxThreadsDim[2]), "Threads");
		gStatus.SetStatisticChanged("Processor", "Max. Grid Size", QString::number(DeviceProperties.maxGridSize[0]) + " x " + QString::number(DeviceProperties.maxGridSize[1]) + " x " + QString::number(DeviceProperties.maxGridSize[2]), "Blocks");
		gStatus.SetStatisticChanged("Processor", "Warp Size", QString::number(DeviceProperties.warpSize), "Threads");
		gStatus.SetStatisticChanged("Processor", "Max. No. Threads/Block", QString::number(DeviceProperties.maxThreadsPerBlock), "Threads");
		gStatus.SetStatisticChanged("Processor", "Max. Shared Memory Per Block", QString::number((float)DeviceProperties.sharedMemPerBlock / 1024.0f, 'f', 2), "KB");
		gStatus.SetStatisticChanged("Processor", "Registers Available Per Block", QString::number((float)DeviceProperties.regsPerBlock / 1024.0f, 'f', 2), "KB");

		// Texture
		gStatus.SetStatisticChanged(DeviceString, "Texture", "", "", "checkerboard");
		gStatus.SetStatisticChanged("Texture", "Max. Dimension Size 1D", QString::number(DeviceProperties.maxTexture1D), "Pixels");
		gStatus.SetStatisticChanged("Texture", "Max. Dimension Size 2D", QString::number(DeviceProperties.maxTexture2D[0]) + " x " + QString::number(DeviceProperties.maxTexture2D[1]), "Pixels");
		gStatus.SetStatisticChanged("Texture", "Max. Dimension Size 3D", QString::number(DeviceProperties.maxTexture3D[0]) + " x " + QString::number(DeviceProperties.maxTexture3D[1]) + " x " + QString::number(DeviceProperties.maxTexture3D[2]), "Pixels");
		gStatus.SetStatisticChanged("Texture", "Alignment", QString::number((float)DeviceProperties.textureAlignment / powf(1024.0f, 2.0f), 'f', 2), "MB");
	}	
	
	return true;
}
*/