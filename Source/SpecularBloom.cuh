#pragma once

#include "Geometry.h"
#include "MonteCarlo.h"
#include "Scene.h"
#include "CudaFrameBuffers.h"

#define KRNL_SP_BLOCK_W		32
#define KRNL_SP_BLOCK_H		8
#define KRNL_SP_BLOCK_SIZE	KRNL_SP_BLOCK_W * KRNL_SP_BLOCK_H

KERNEL void KrnlSpecularBloom(int* pSeeds)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int PID	= Y * gFilmWidth + X;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	CRNG RNG(&pSeeds[2 * PID], &pSeeds[2 * PID + 1]);

	CColorXyz Lb = SPEC_BLACK;

	float SumWeight = 0.0f;

	float Radius = 30.0f;

	for (int i = 0; i < 5; i++)
	{
		Vec2f C = Radius * ConcentricSampleDisk(RNG.Get2());

//		Vec2f UV(X + C.x, Y + C.y);
		Vec2f UV((float)X + C.x, (float)Y + C.y);

		if (UV.x < 0.0f || UV.x >= gFilmWidth || UV.y < 0.0f || UV.y >= gFilmHeight)
			break;
		
		const float4 Le = tex2D(gTexRunningEstimateXyza, UV.x, UV.y);

		Lb += expf(-5.0f * (C.Length() / Radius)) * CColorXyz(Le.x, Le.y, Le.z);
		SumWeight++;
	}

	__syncthreads();

	if (SumWeight > 0.0f)
		Lb /= SumWeight;

	float4 ColorXyza = tex2D(gTexRunningSpecularBloomXyza, X, Y) + (make_float4(Lb.c[0], Lb.c[1], Lb.c[2], 0.0f) - tex2D(gTexRunningSpecularBloomXyza, X, Y)) / (float)__max(1.0f, gNoIterations);

	surf2Dwrite(ColorXyza, gSurfRunningSpecularBloomXyza, X * sizeof(float4), Y);
}

void SpecularBloom(CScene& Scene, CScene* pDevScene, int* pSeeds, CCudaFrameBuffers& CudaFrameBuffers)
{
	const dim3 KernelBlock(KRNL_SP_BLOCK_W, KRNL_SP_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)Scene.m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)Scene.m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));

	KrnlSpecularBloom<<<KernelGrid, KernelBlock>>>(pSeeds);
	HandleCudaError(cudaGetLastError());
}