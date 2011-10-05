#pragma once

#include "Transport.cuh"

KERNEL void KrnlMultipleScattering(CScene* pScene, int* pSeeds, CColorXyz* pDevEstFrameXyz)
{
	const int X = (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global y
	const int Y	= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global x
	
	// Compute sample ID
	const int SID = (Y * (gridDim.x * blockDim.x)) + X;

	// Exit if beyond kernel boundaries
	if (X >= pScene->m_Camera.m_Film.m_Resolution.GetResX() || Y >= pScene->m_Camera.m_Film.m_Resolution.GetResY())
		return;
	
	// Init random number generator
	CRNG RNG(&pSeeds[SID * 2], &pSeeds[SID * 2 + 1]);

	CColorXyz Lv = SPEC_BLACK, Li = SPEC_BLACK;

	CRay Re;

 	// Generate the camera ray
 	pScene->m_Camera.GenerateRay(Vec2f(X, Y), RNG.Get2(), Re.m_O, Re.m_D);

	Re.m_MinT = 0.0f; 
	Re.m_MaxT = FLT_MAX;

	Vec3f Pe, Pl, Normal;
	
	CLight* pLight = NULL;

	if (SampleDistanceRM(Re, RNG, Pe, pScene))
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, (Pe - Re.m_O).Length()), Li, Pl, pLight))
		{
			pDevEstFrameXyz[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = Li;
			return;
		}

		// Fetch density
		const float D = GetDensity(pScene, Pe);

		// Get opacity at eye point
		const CColorXyz	Ke = GetEmission(pScene, D).ToXYZ();
		
		// Estimate direct light at eye point
	 	Lv = Ke + UniformSampleOneLight(pScene, Normalize(-Re.m_D), Pe, NormalizedGradient(pScene, Pe), RNG, false);
	}
	else
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, INF_MAX), Li, Pl, pLight))
			Lv = Li;
	}

	// Contribute
	pDevEstFrameXyz[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = Lv;
}

void MultipleScattering(CScene* pScene, CScene* pDevScene, int* pSeeds, CColorXyz* pDevEstFrameXyz)
{
	const dim3 KernelBlock(pScene->m_KernelSize.x, pScene->m_KernelSize.y);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));
	
	KrnlMultipleScattering<<<KernelGrid, KernelBlock>>>(pDevScene, pSeeds, pDevEstFrameXyz);
}