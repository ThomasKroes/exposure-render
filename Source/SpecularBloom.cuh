#pragma once

#include "Geometry.h"
#include "Scene.h"

#define KRNL_SP_BLOCK_W		32
#define KRNL_SP_BLOCK_H		8
#define KRNL_SP_BLOCK_SIZE	KRNL_SP_BLOCK_W * KRNL_SP_BLOCK_H

KERNEL void KrnlSpecularBloom(CColorXyz* pImage, CColorXyz* pTempImage, int* pSeeds)
{
	const int X 	= (blockIdx.x * blockDim.x) + threadIdx.x;
	const int Y		= (blockIdx.y * blockDim.y) + threadIdx.y;
	const int PID	= (Y * gFilmWidth) + X;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	CRNG RNG(&pSeeds[2 * PID], &pSeeds[2 * PID + 1]);

//	CColorXyz L = ;

//	pImage[PID]	= Sum / SumW;
}

void SpecularBloom(CScene& Scene, CScene* pDevScene, int* pSeeds)
{
	/*
	const dim3 KernelBlock(KRNL_SP_BLOCK_W, KRNL_SP_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)Scene.m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)Scene.m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));

	KrnlSpecularBloom<<<KernelGrid, KernelBlock>>>(pImage, pTempImage); 
	KrnlSpecularBloom<<<KernelGrid, KernelBlock>>>(pImage, pTempImage); 
	*/
}