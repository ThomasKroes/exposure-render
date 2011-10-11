#pragma once

#include <cuda_runtime.h>
#include <cutil.h>

#include "CudaUtilities.h"

#define KRNL_NI_BLOCK_W		1
#define KRNL_NI_BLOCK_H		1
#define KRNL_NI_BLOCK_SIZE	KRNL_NI_BLOCK_W * KRNL_NI_BLOCK_H

KERNEL void KrnlNearestIntersection(CScene* pScene, float* pT)
{
	CRay Rc;
	
	const Vec2f UV(0.5f * (float)gFilmWidth, 0.5f * (float)gFilmHeight);

	pScene->m_Camera.GenerateRay(UV, Vec2f(0.0f), Rc.m_O, Rc.m_D);

	Rc.m_MinT = 0.0f;
	Rc.m_MaxT = INF_MAX;

	NearestIntersection(Rc, pScene, *pT);
}

float NearestIntersection(CScene* pDevScene)
{
	const dim3 KernelBlock(KRNL_NI_BLOCK_W, KRNL_NI_BLOCK_H);
	const dim3 KernelGrid(1, 1);
	
	float T = 0.0f;

	float* pDevT = NULL;

	HandleCudaError(cudaMalloc(&pDevT, sizeof(float)));

	KrnlNearestIntersection<<<KernelGrid, KernelBlock>>>(pDevScene, pDevT);
	HandleCudaError(cudaGetLastError());
	cudaThreadSynchronize();

	HandleCudaError(cudaMemcpy(&T, pDevT, sizeof(float), cudaMemcpyDeviceToHost));
	HandleCudaError(cudaFree(pDevT));

	return T;
}