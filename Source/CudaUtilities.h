#pragma once

#include <cutil_inline.h>

class CCudaTimer
{
public:
	CCudaTimer(void);
	virtual ~CCudaTimer(void);

	void	StartTimer(void);
	float	StopTimer(void);

private:
	bool			m_Started;
	cudaEvent_t 	m_EventStart;
	cudaEvent_t 	m_EventStop;
};

bool InitializeCuda(void);
void HandleCudaError(const cudaError_t CudaError);