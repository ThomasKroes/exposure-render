#pragma once

#include "Geometry.h"
#include "Filter.h"
#include "Scene.h"
#include "CudaUtilities.h"

#define KRNL_BLUR_BLOCK_W		16
#define KRNL_BLUR_BLOCK_H		8
#define KRNL_BLUR_BLOCK_SIZE	KRNL_BLUR_BLOCK_W * KRNL_BLUR_BLOCK_H

KERNEL void KrnlBlurXyzH(CCudaView* pView)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int PID	= Y * gFilmWidth + X;
	const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	const int X0 = max((int)ceilf(X - gFilterWidth), 0);
	const int X1 = min((int)floorf(X + gFilterWidth), (int)gFilmWidth - 1);

	CColorXyza Sum;

	__shared__ float FW[KRNL_BLUR_BLOCK_SIZE];
	__shared__ float SumW[KRNL_BLUR_BLOCK_SIZE];

	__syncthreads();

	FW[TID]		= 0.0f;
	SumW[TID]	= 0.0f;

	for (int x = X0; x <= X1; x++)
	{
		FW[TID] = gFilterWeights[(int)fabs((float)x - X)];

		Sum			+= FW[TID] * pView->m_EstimateFrameXyza.m_pData[Y * gFilmWidth + x];
		SumW[TID]	+= FW[TID];
	}

	Sum /= SumW[TID];

	pView->m_FrameBlurXyza.m_pData[PID].c[0] = Sum.c[0];
	pView->m_FrameBlurXyza.m_pData[PID].c[1] = Sum.c[1];
	pView->m_FrameBlurXyza.m_pData[PID].c[2] = Sum.c[2];
}

KERNEL void KrnlBlurXyzV(CCudaView* pView)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int PID	= Y * gFilmWidth + X;
	const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	const int Y0 = max((int)ceilf (Y - gFilterWidth), 0);
	const int Y1 = min((int)floorf(Y + gFilterWidth), gFilmHeight - 1);

	CColorXyza Sum;

	__shared__ float FW[KRNL_BLUR_BLOCK_SIZE];
	__shared__ float SumW[KRNL_BLUR_BLOCK_SIZE];

	__syncthreads();

	FW[TID]		= 0.0f;
	SumW[TID]	= 0.0f;

	for (int y = Y0; y <= Y1; y++)
	{
		FW[TID] = gFilterWeights[(int)fabs((float)y - Y)];

		Sum			+= FW[TID] * pView->m_FrameBlurXyza.m_pData[y * gFilmWidth + X];
		SumW[TID]	+= FW[TID];
	}

	Sum /= SumW[TID];

	const float4 ColorXYZA = make_float4(Sum.c[0], Sum.c[1], Sum.c[2], 0.0f);

	pView->m_EstimateFrameXyza.m_pData[PID].c[0] = Sum.c[0];
	pView->m_EstimateFrameXyza.m_pData[PID].c[1] = Sum.c[1];
	pView->m_EstimateFrameXyza.m_pData[PID].c[2] = Sum.c[2];
}

void BlurImageXyz(CScene* pScene, CScene* pDevScene, CCudaView* pDevView)
{
	const dim3 KernelBlock(KRNL_BLUR_BLOCK_W, KRNL_BLUR_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));

	KrnlBlurXyzH<<<KernelGrid, KernelBlock>>>(pDevView);
	cudaThreadSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Blur Estimate H");
	
	KrnlBlurXyzV<<<KernelGrid, KernelBlock>>>(pDevView);
	cudaThreadSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Blur Estimate V");
}