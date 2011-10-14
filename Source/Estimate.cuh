#pragma once

#include "Geometry.h"

#define KRNL_ESTIMATE_BLOCK_W		8
#define KRNL_ESTIMATE_BLOCK_H		8
#define KRNL_ESTIMATE_BLOCK_SIZE	KRNL_ESTIMATE_BLOCK_W * KRNL_ESTIMATE_BLOCK_H

KERNEL void KrnlEstimate(CCudaView* pView)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int PID	= Y * gFilmWidth + X;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	pView->m_RunningEstimateXyza.m_pData[PID] = CumulativeMovingAverage(pView->m_RunningEstimateXyza.m_pData[PID], pView->m_FrameEstimateXyza.m_pData[PID], gNoIterations);
}

void Estimate(CScene* pScene, CScene* pDevScene, CCudaView* pDevView)
{
	const dim3 KernelBlock(KRNL_ESTIMATE_BLOCK_W, KRNL_ESTIMATE_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.GetWidth() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.GetHeight() / (float)KernelBlock.y));

	KrnlEstimate<<<KernelGrid, KernelBlock>>>(pDevView);
	cudaThreadSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Compute Estimate");
}