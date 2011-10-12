#pragma once

#include "Geometry.h"
#include "Variance.h"
#include "CudaFrameBuffers.h"

#define KRNL_ESTIMATE_BLOCK_W		16
#define KRNL_ESTIMATE_BLOCK_H		8
#define KRNL_ESTIMATE_BLOCK_SIZE	KRNL_ESTIMATE_BLOCK_W * KRNL_ESTIMATE_BLOCK_H

KERNEL void KrnlEstimate(CCudaView* pView)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int PID	= Y * gFilmWidth + X;
	const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	pView->m_EstimateXyza.m_pData[PID] = gNoIterations == 0 ? CColorXyza(0.0f) : pView->m_EstimateXyza.m_pData[Y * gFilmWidth + X] + (pView->m_EstimateFrameXyza.m_pData[Y * gFilmWidth + X] - pView->m_EstimateXyza.m_pData[Y * gFilmWidth + X] / gNoIterations);
}

void Estimate(CScene* pScene, CScene* pDevScene, CCudaView* pDevView)
{
	const dim3 KernelBlock(KRNL_ESTIMATE_BLOCK_W, KRNL_ESTIMATE_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.GetWidth() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.GetHeight() / (float)KernelBlock.y));

	KrnlEstimate<<<KernelGrid, KernelBlock>>>(pDevView);
	cudaThreadSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Compute Estimate");
}