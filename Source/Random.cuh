#pragma once

#include "Scene.h"

#include "curand_kernel.h"

class CScene;

KERNEL void KrnlSetupRNG(CScene* pScene, curandStateXORWOW_t* pDevRandomStates)
{
	const int X		= (blockIdx.x * blockDim.x) + threadIdx.x;
	const int Y		= (blockIdx.y * blockDim.y) + threadIdx.y;

	// Exit if beyond canvas boundaries
	if (X >= pScene->m_Camera.m_Film.m_Resolution.GetResX() || Y >= pScene->m_Camera.m_Film.m_Resolution.GetResY())
		return;

	// Initialize
	curand_init(Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X, 1234, 0, &pDevRandomStates[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResY() + X]);
}

extern "C" void SetupRNG(CScene* pScene, CScene* pDevScene, curandStateXORWOW_t* pDevRandomStates)
{
	const dim3 KernelBlock(32, 8);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));

	KrnlSetupRNG<<<KernelGrid, KernelBlock>>>(pDevScene, pDevRandomStates);

	cudaError_t Error = cudaGetLastError();
}