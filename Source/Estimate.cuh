#pragma once

#include "Geometry.h"
#include "Variance.h"
#include "CudaFrameBuffers.h"

#define KRNL_ESTIMATE_BLOCK_W		32
#define KRNL_ESTIMATE_BLOCK_H		8
#define KRNL_ESTIMATE_BLOCK_SIZE	KRNL_ESTIMATE_BLOCK_W * KRNL_ESTIMATE_BLOCK_H

KERNEL void KrnlEstimate(float N)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int PID	= Y * gFilmWidth + X;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	float4 ColorXyza = tex2D(gTexRunningEstimateXyza, X, Y) + (tex2D(gTexFrameEstimateXyza, X, Y) - tex2D(gTexRunningEstimateXyza, X, Y)) / (float)__max(1.0f, N);
	
	surf2Dwrite(ColorXyza, gSurfRunningEstimateXyza, X * sizeof(float4), Y);
}

void Estimate(CScene* pScene, CScene* pDevScene, CCudaFrameBuffers& CudaFrameBuffers, float N)
{
	const dim3 KernelBlock(KRNL_ESTIMATE_BLOCK_W, KRNL_ESTIMATE_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.GetWidth() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.GetHeight() / (float)KernelBlock.y));

	KrnlEstimate<<<KernelGrid, KernelBlock>>>(N); 
}