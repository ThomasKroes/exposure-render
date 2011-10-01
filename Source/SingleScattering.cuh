#pragma once

#include "Transport.cuh"

#include <algorithm>
#include <vector>

DEV CColorXyz IncidentLight(CScene* pScene, const Vec3f& Wo, const Vec3f& Pe, const float& D, CCudaRNG& RNG)
{
	switch (pScene->m_ShadingType)
	{
		// Do BRDF shading
		case 0:
		{
			return UniformSampleOneLight(pScene, Normalize(Wo), Pe, NormalizedGradient(pScene, Pe), RNG, false);
		}
		
		// Do phase function shading
		case 1:
		{
			return 0.5f * UniformSampleOneLight(pScene, Normalize(Wo), Pe, NormalizedGradient(pScene, Pe), RNG, false);
		}

		// Do hybrid shading (BRDF + phase function)
		case 2:
		{
			// Get normalized gradient magnitude
			const float GradientMagnitude = GetGradientMagnitude(pScene, Pe) / pScene->m_GradientMagnitudeRange.GetLength();

			// Determine BRDF vs Phase Function scattering
			const float PdfBrdf = 1.0f - __expf(-pScene->m_GradientFactor * GradientMagnitude);

			if (RNG.Get1() < PdfBrdf)
			{
  				return 2.0f * UniformSampleOneLight(pScene, Normalize(Wo), Pe, NormalizedGradient(pScene, Pe), RNG, true);
			}
			else
			{
				return UniformSampleOneLight(pScene, Normalize(Wo), Pe, NormalizedGradient(pScene, Pe), RNG, false);
			}
		}
	}

	return SPEC_BLACK;
}

KERNEL void KrnlSingleScattering(CScene* pScene, unsigned int* pSeeds, CColorXyz* pDevEstFrameXyz)
{
	const int X = (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global y
	const int Y	= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global x
	
	// Compute sample ID
	const int SID = (Y * (gridDim.x * blockDim.x)) + X;

	// Exit if beyond kernel boundaries
	if (X >= pScene->m_Camera.m_Film.GetWidth() || Y >= pScene->m_Camera.m_Film.GetHeight())
		return;
	
	// Init random number generator
	CCudaRNG RNG(&pSeeds[SID * 2], &pSeeds[SID * 2 + 1]);

	CColorXyz Lv = SPEC_BLACK, Li = SPEC_BLACK;

	CRay Re;

 	// Generate the camera ray
 	pScene->m_Camera.GenerateRay(Vec2f(X, Y), RNG.Get2(), Re.m_O, Re.m_D);

	Re.m_MinT = 0.0f; 
	Re.m_MaxT = FLT_MAX;

	Vec3f Pe, Pl;
	
	CLight* pLight = NULL;

	if (SampleDistanceRM(Re, RNG, Pe, pScene))
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, (Pe - Re.m_O).Length()), Li, Pl, pLight))
		{
			pDevEstFrameXyz[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = Li;
			return;
		}

		// Fetch density
		const float D = Density(pScene, Pe);

		// Add emission
		Lv += GetEmission(pScene, D).ToXYZ();

		// Add incident light
		Lv += IncidentLight(pScene, -Re.m_D, Pe, D, RNG);
	}
	else
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, INF_MAX), Li, Pl, pLight))
			Lv = Li;
	}

	pDevEstFrameXyz[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = Lv;
}

void SingleScattering(CScene* pScene, CScene* pDevScene, unsigned int* pSeeds, CColorXyz* pDevEstFrameXyz)
{
	const dim3 KernelBlock(16, 8);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));
	
	KrnlSingleScattering<<<KernelGrid, KernelBlock>>>(pDevScene, pSeeds, pDevEstFrameXyz);
}