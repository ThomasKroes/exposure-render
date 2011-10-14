#pragma once

#include "Transport.cuh"

#define KRNL_SS_BLOCK_W		16
#define KRNL_SS_BLOCK_H		8
#define KRNL_SS_BLOCK_SIZE	KRNL_SS_BLOCK_W * KRNL_SS_BLOCK_H

KERNEL void KrnlSingleScattering(CScene* pScene, CCudaView* pView)
{
	const int X		= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int PID	= Y * gFilmWidth + X;
	const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

	if (X >= gFilmWidth || Y >= gFilmHeight || PID >= gFilmNoPixels)
		return;
	
	CRNG RNG(&pView->m_RandomSeeds1.m_pData[PID], &pView->m_RandomSeeds2.m_pData[PID]);

	CColorXyz Lv = SPEC_BLACK, Li = SPEC_BLACK;

	CRay Re;
	
	const Vec2f UV = Vec2f(X, Y) + RNG.Get2();

 	pScene->m_Camera.GenerateRay(UV, RNG.Get2(), Re.m_O, Re.m_D);

	Re.m_MinT = 0.0f; 
	Re.m_MaxT = 15000.0f;

	Vec3f Pe, Pl;
	
	CLight* pLight = NULL;

	/*
	if (SampleDistanceRM(Re, RNG, Pe))
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, (Pe - Re.m_O).Length()), Li, Pl, pLight))
		{
			pView->m_FrameEstimateXyza.m_pData[PID].c[0] = Lv.c[0];
			pView->m_FrameEstimateXyza.m_pData[PID].c[1] = Lv.c[1];
			pView->m_FrameEstimateXyza.m_pData[PID].c[2] = Lv.c[2];

			return;
		}

		const float D = GetNormalizedIntensity(Pe);

		Lv += GetEmission(D).ToXYZ();

		switch (pScene->m_ShadingType)
		{
			case 0:
			{
				Lv += UniformSampleOneLight(pScene, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe), RNG, true);
				break;
			}
		
			case 1:
			{
				Lv += 0.5f * UniformSampleOneLight(pScene, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe), RNG, false);
				break;
			}

			case 2:
			{
				const float GradMag = GradientMagnitude(Pe) * gIntensityInvRange;

				const float PdfBrdf = (1.0f - __expf(-pScene->m_GradientFactor * GradMag));

				if (RNG.Get1() < PdfBrdf)
  					Lv += UniformSampleOneLight(pScene, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe), RNG, true);
				else
 					Lv += 0.5f * UniformSampleOneLight(pScene, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe), RNG, false);

				break;
			}
		}
	}
	else
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, INF_MAX), Li, Pl, pLight))
			Lv = Li;
	}
	*/

	pView->m_FrameEstimateXyza.m_pData[PID].c[0] = Lv.c[0];
	pView->m_FrameEstimateXyza.m_pData[PID].c[1] = Lv.c[1];
	pView->m_FrameEstimateXyza.m_pData[PID].c[2] = Lv.c[2];
}

void SingleScattering(CScene* pScene, CScene* pDevScene, CCudaView* pView)
{
	const dim3 KernelBlock(KRNL_SS_BLOCK_W, KRNL_SS_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));
	
	KrnlSingleScattering<<<KernelGrid, KernelBlock>>>(pDevScene, pView);
	cudaThreadSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Single Scattering");
}