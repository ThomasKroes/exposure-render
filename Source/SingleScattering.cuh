#pragma once

#include "Transport.cuh"

#include <algorithm>
#include <vector>

KERNEL void KrnlSingleScattering(CScene* pScene, int* pSeeds, CColorXyz* pDevEstFrameXyz)
{
	const int X = (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global y
	const int Y	= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global x
	
	// Compute pixel ID
	const int PID = (Y * pScene->m_Camera.m_Film.GetWidth()) + X;

	// Exit if beyond render canvas boundaries
	if (X >= pScene->m_Camera.m_Film.GetWidth() || Y >= pScene->m_Camera.m_Film.GetHeight() || PID >= pScene->m_Camera.m_Film.GetWidth() * pScene->m_Camera.m_Film.GetHeight())
		return;
	
	// Init random number generator
	CRNG RNG(&pSeeds[2 * PID], &pSeeds[2 * PID + 1]);

	CColorXyz Lv = SPEC_BLACK, Li = SPEC_BLACK;

	CRay Re;
	
	// Generate the camera ray
 	pScene->m_Camera.GenerateRay(Vec2f(X, Y) + RNG.Get2(), RNG.Get2(), Re.m_O, Re.m_D);

	Re.m_MinT = 0.0f; 
	Re.m_MaxT = FLT_MAX;

	Vec3f Pe, Pl;
	
	CLight* pLight = NULL;

	if (SampleDistanceRM(Re, RNG, Pe, pScene))
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, (Pe - Re.m_O).Length()), Li, Pl, pLight))
		{
			pDevEstFrameXyz[PID] = Li;
			return;
		}
		 
		const float D = GetDensity(pScene, Pe);

		const CColorXyz	Ke = GetEmission(pScene, D).ToXYZ();
		
		Lv += Ke;

		switch (pScene->m_ShadingType)
		{
			// Do BRDF shading
			case 0:
			{
				Lv += UniformSampleOneLight(pScene, Normalize(-Re.m_D), Pe, NormalizedGradient(pScene, Pe), RNG, true);
				break;
			}
		
			// Do phase function shading
			case 1:
			{
				Lv += 0.5f * UniformSampleOneLight(pScene, Normalize(-Re.m_D), Pe, NormalizedGradient(pScene, Pe), RNG, false);
				break;
			}

			// Do hybrid shading (BRDF + phase function)
			case 2:
			{
				const float GradMag = GradientMagnitude(pScene, Pe) / pScene->m_GradientMagnitudeRange.GetLength();

				const float PdfBrdf = (1.0f - __expf(-pScene->m_GradientFactor * GradMag));

				if (RNG.Get1() < PdfBrdf)
  					Lv += UniformSampleOneLight(pScene, Normalize(-Re.m_D), Pe, NormalizedGradient(pScene, Pe), RNG, true);
				else
 					Lv += 0.5f * UniformSampleOneLight(pScene, Normalize(-Re.m_D), Pe, NormalizedGradient(pScene, Pe), RNG, false);

				break;
			}
		}
	}
	else
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, INF_MAX), Li, Pl, pLight))
			Lv = Li;
	}
	
	pDevEstFrameXyz[PID] = Lv;
}

void SingleScattering(CScene* pScene, CScene* pDevScene, int* pSeeds, CColorXyz* pDevEstFrameXyz)
{
	const dim3 KernelBlock(16, 8);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));
	
	KrnlSingleScattering<<<KernelGrid, KernelBlock>>>(pDevScene, pSeeds, pDevEstFrameXyz);
}