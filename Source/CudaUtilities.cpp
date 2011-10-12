
// Precompiled headers
#include "Stable.h"

#include "CudaUtilities.h"

CCudaTimer::CCudaTimer(void)
{
	StartTimer();
}

CCudaTimer::~CCudaTimer(void)
{
	cudaEventDestroy(m_EventStart);
	cudaEventDestroy(m_EventStop);
}

void CCudaTimer::StartTimer(void)
{
	cudaEventCreate(&m_EventStart);
	cudaEventCreate(&m_EventStop);
	cudaEventRecord(m_EventStart, 0);

	m_Started = true;
}

float CCudaTimer::StopTimer(void)
{
	if (!m_Started)
		return 0.0f;

	cudaEventRecord(m_EventStop, 0);
	cudaEventSynchronize(m_EventStop);

	float TimeDelta = 0.0f;

	cudaEventElapsedTime(&TimeDelta, m_EventStart, m_EventStop);
	cudaEventDestroy(m_EventStart);
	cudaEventDestroy(m_EventStop);

	m_Started = false;

	return TimeDelta;
}

float CCudaTimer::ElapsedTime(void)
{
	if (!m_Started)
		return 0.0f;

	cudaEventRecord(m_EventStop, 0);
	cudaEventSynchronize(m_EventStop);

	float TimeDelta = 0.0f;

	cudaEventElapsedTime(&TimeDelta, m_EventStart, m_EventStop);

	m_Started = false;

	return TimeDelta;
}

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void GetCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
	CUresult error = 	cuDeviceGetAttribute( attribute, device_attribute, device );

	if( CUDA_SUCCESS != error) {
		fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
			error, __FILE__, __LINE__);
		exit(-1);
	}
}

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
	
	return cudaSetDevice(cutGetMaxGflopsDeviceId()) == cudaSuccess;
}

void HandleCudaError(const cudaError_t CudaError, const char* pDescription /*= ""*/)
{
	if (CudaError == cudaSuccess)
		return;

	throw new QString("Encountered a critical CUDA error: " + QString::fromAscii(pDescription) + " " + QString(cudaGetErrorString(CudaError)));
}

void HandleCudaKernelError(const cudaError_t CudaError, const char* pName /*= ""*/)
{
	if (CudaError == cudaSuccess)
		return;

	throw new QString("The '" + QString::fromAscii(pName) + "' kernel caused the following CUDA runtime error: " + QString(cudaGetErrorString(CudaError)));
}

int GetTotalCudaMemory(void)
{
	size_t Available = 0, Total = 0;
	cudaMemGetInfo(&Available, &Total);
	return Total;
}

int GetAvailableCudaMemory(void)
{
	size_t Available = 0, Total = 0;
	cudaMemGetInfo(&Available, &Total);
	return Available;
}

int GetUsedCudaMemory(void)
{
	size_t Available = 0, Total = 0;
	cudaMemGetInfo(&Available, &Total);
	return Total - Available;
}