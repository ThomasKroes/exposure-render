
#include <host_defines.h>
#include <float.h>
#include <algorithm>
#include <math.h>

using namespace std;

#define KERNEL					__global__
#define HOST					__host__
#define DEVICE					__device__
#define HOST_DEVICE				HOST DEVICE
#define HOD						__host__ __device__
#define CD						__device__ __constant__
#define	DNI						__device__ __noinline__
#define	HODNI					__host__ __device__ __noinline__
#define PI_F					3.141592654f	
#define HALF_PI_F				0.5f * PI_F
#define QUARTER_PI_F			0.25f * PI_F
#define TWO_PI_F				2.0f * PI_F
#define INV_PI_F				0.31830988618379067154f
#define INV_TWO_PI_F			0.15915494309189533577f
#define FOUR_PI_F				4.0f * PI_F
#define INV_4_PI_F				1.0f / FOUR_PI_F
#define	EULER_F					2.718281828f
#define RAD_F					57.29577951308232f
#define TWO_RAD_F				2.0f * RAD_F
#define DEG_TO_RAD				1.0f / RAD_F

/*
HOST_DEVICE inline void HandleCudaError(const cudaError_t CudaError, const char* pDescription = "")
{
	if (CudaError == cudaSuccess)
		return;

//	Log(QString("Encountered a critical CUDA error: " + QString::fromAscii(pDescription) + " " + QString(cudaGetErrorString(CudaError))));

//	throw new QString("Encountered a critical CUDA error: " + QString::fromAscii(pDescription) + " " + QString(cudaGetErrorString(CudaError)));
}

HOST_DEVICE inline void HandleCudaKernelError(const cudaError_t CudaError, const char* pName = "")
{
	if (CudaError == cudaSuccess)
		return;

//	Log(QString("The '" + QString::fromAscii(pName) + "' kernel caused the following CUDA runtime error: " + QString(cudaGetErrorString(CudaError))));

//	throw new QString("The '" + QString::fromAscii(pName) + "' kernel caused the following CUDA runtime error: " + QString(cudaGetErrorString(CudaError)));
}
*/