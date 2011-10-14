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