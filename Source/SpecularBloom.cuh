#pragma once

#include "Geometry.h"
#include "MonteCarlo.h"
#include "Scene.h"
#include "CudaFrameBuffers.h"

#define KRNL_SP_BLOCK_W		32
#define KRNL_SP_BLOCK_H		8
#define KRNL_SP_BLOCK_SIZE	KRNL_SP_BLOCK_W * KRNL_SP_BLOCK_H

KERNEL void KrnlSpecularBloom(int* pSeeds, CColorXyz* pEstXyz, CColorXyz* pSpecularBloom, int N)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int PID	= Y * gFilmWidth + X;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	CRNG RNG(&pSeeds[2 * PID], &pSeeds[2 * PID + 1]);

	CColorXyz Lb = SPEC_BLACK;

	float SumWeight = 0.0f;

	for (int i = 0; i < 20; i++)
	{
		Vec2f C = 100.0f * ConcentricSampleDisk(RNG.Get2());

//		Vec2f UV(X + C.x, Y + C.y);
		Vec2f UV((float)X + C.x, (float)Y + C.y);

		if (UV.x < 0.0f || UV.x >= gFilmWidth || UV.y < 0.0f || UV.y >= gFilmHeight)
			break;
		
		const int PIDB = floorf(UV.y) * gFilmWidth + UV.x;

		Lb += expf(-4.0f * (C.Length() / 100.0f)) * pEstXyz[PIDB];//expf(-(C.Length() / 30.0f)) * 10.0f * 
		SumWeight++;
	}

	__syncthreads();

	if (SumWeight > 0.0f)
		pSpecularBloom[PID] += Lb / SumWeight;

//	pEstXyz[PID] += pSpecularBloom[PID] / (float)__max(1.0f, N);
}

void SpecularBloom(CScene& Scene, CScene* pDevScene, int* pSeeds, CCudaFrameBuffers& CudaFrameBuffers, int N)
{
	const dim3 KernelBlock(KRNL_SP_BLOCK_W, KRNL_SP_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)Scene.m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)Scene.m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));

	KrnlSpecularBloom<<<KernelGrid, KernelBlock>>>(pSeeds, CudaFrameBuffers.m_pDevEstXyz, CudaFrameBuffers.m_pDevSpecularBloom, N); 
}