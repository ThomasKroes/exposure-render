#pragma once

#include "Transport.cuh"
#include "CudaUtilities.h"

#define KRNL_MS_BLOCK_W		32
#define KRNL_MS_BLOCK_H		8
#define KRNL_MS_BLOCK_SIZE	KRNL_MS_BLOCK_W * KRNL_MS_BLOCK_H

KERNEL void KrnlMultipleScattering(CScene* pScene, int* pSeeds)
{
	const int X		= (blockIdx.x * blockDim.x) + threadIdx.x;
	const int Y		= (blockIdx.y * blockDim.y) + threadIdx.y;
	const int PID	= (Y * gFilmWidth) + X;

	if (X >= gFilmWidth || Y >= gFilmHeight || PID >= gFilmNoPixels)
		return;
	
	CRNG RNG(&pSeeds[2 * PID], &pSeeds[2 * PID + 1]);

	CColorXyz Lv = SPEC_BLACK, Li = SPEC_BLACK, Tr = SPEC_WHITE;

	CRay Re;
	
	const Vec2f UV = Vec2f(X, Y) + RNG.Get2();

 	pScene->m_Camera.GenerateRay(UV, RNG.Get2(), Re.m_O, Re.m_D);

	Re.m_MinT = 0.0f; 
	Re.m_MaxT = FLT_MAX;

	Vec3f Pe, Pl;
	
	CLight* pLight = NULL;

	for (int i = 0; i < 2; i++)
	{
		if (SampleDistanceRM(Re, RNG, Pe, pScene))
		{
			if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, (Pe - Re.m_O).Length()), Li, Pl, pLight))
			{
				float4 ColorXYZA = make_float4(Lv.c[0], Lv.c[1], Lv.c[2], 0.0f);
				surf2Dwrite(ColorXYZA, gSurfRunningEstimateXyza, X * sizeof(float4), Y);
				return;
			}
		 
			const float D = GetNormalizedIntensity(pScene, Pe);

			Lv += Tr * GetEmission(pScene, D).ToXYZ();

			Lv += Tr * 0.5f * UniformSampleOneLight(pScene, D, Normalize(-Re.m_D), Pe, NormalizedGradient(pScene, Pe), RNG, false);
		}
		else
		{
			if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, INF_MAX), Li, Pl, pLight))
				Lv += Tr * Li;

			break;
		}

		Re.m_O		= Pe;
		Re.m_D		= UniformSampleSphere(RNG.Get2());
		Re.m_MinT	= 0.0f;
		Re.m_MaxT	= INF_MAX;

		Tr *= INV_4_PI_F;
	}

	__syncthreads();

	float4 ColorXYZA = make_float4(Lv.c[0], Lv.c[1], Lv.c[2], 0.0f);
	surf2Dwrite(ColorXYZA, gSurfRunningEstimateXyza, X * sizeof(float4), Y);
}

void MultipleScattering(CScene* pScene, CScene* pDevScene, int* pSeeds)
{
	const dim3 KernelBlock(KRNL_MS_BLOCK_W, KRNL_MS_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));
	
	KrnlMultipleScattering<<<KernelGrid, KernelBlock>>>(pDevScene, pSeeds);
	cudaThreadSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Multiple Scattering");
}