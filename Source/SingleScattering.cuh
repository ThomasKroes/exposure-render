#pragma once

#include "Geometry.h"

#include "Transport.cuh"

// Find the nearest non-empty voxel in the volume
DEV inline bool NearestIntersection(CScene* pScene, CRay& R, const float& StepSize, const float& U, float* pBoxMinT = NULL, float* pBoxMaxT = NULL)
{
	float MinT;
	float MaxT;

	// Intersect the eye ray with bounding box, if it does not intersect then return the environment
	if (!pScene->m_BoundingBox.Intersect(R, &MinT, &MaxT))
		return false;

	bool Hit = false;

	if (pBoxMinT)
		*pBoxMinT = MinT;

	if (pBoxMaxT)
		*pBoxMaxT = MaxT;

	MinT += U * StepSize;

	// Step through the volume and stop as soon as we come across a non-empty voxel
	while (MinT < MaxT)
	{
		if (GetOpacity(pScene, Density(pScene, R(MinT))).r > 0.0f)
		{
			Hit = true;
			break;
		}
		else
		{
			MinT += StepSize;
		}
	}

	if (Hit)
	{
		R.m_MinT = MinT;
		R.m_MaxT = MaxT;
	}

	return Hit;
}

// Trace volume with single scattering
KERNEL void KrnlSS(CScene* pScene, unsigned int* pSeeds, CColorXyz* pDevEstFrameXyz)
{
	const int X = (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global y
	const int Y	= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global x
	
	// Compute sample ID
	const int SID = (Y * (gridDim.x * blockDim.x)) + X;

	// Exit if beyond kernel boundaries
	if (X >= pScene->m_Camera.m_Film.m_Resolution.GetResX() || Y >= pScene->m_Camera.m_Film.m_Resolution.GetResY())
		return;
	
	// Init random number generator
	CCudaRNG RNG(&pSeeds[SID * 2], &pSeeds[SID * 2 + 1]);

	CColorXyz Lv = SPEC_BLACK, Li = SPEC_BLACK;

	CRay Re;

 	// Generate the camera ray
 	pScene->m_Camera.GenerateRay(Vec2f(X, Y), RNG.Get2(), Re.m_O, Re.m_D);

	Re.m_MinT = 0.0f; 
	Re.m_MaxT = FLT_MAX;

	Vec3f Pe, Normal;
	
	if (SampleDistanceRM(Re, RNG, Pe, pScene, 0))
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, (Pe - Re.m_O).Length()), Li))
		{
			pDevEstFrameXyz[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = Li;
			return;
		}

		// Fetch density
		const float D = Density(pScene, Pe);

		// Get opacity at eye point
		const float		Tr = GetOpacity(pScene, D).r;
		const CColorXyz	Ke = GetEmission(pScene, D).ToXYZ();
		
		// Estimate direct light at eye point
	 	Lv = Ke + UniformSampleOneLight(pScene, Normalize(-Re.m_D), Pe, NormalizedGradient(pScene, Pe), RNG, 0.001f);
	}
	else
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, INF_MAX), Li))
			Lv = Li;
	}

	// Contribute
	pDevEstFrameXyz[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = Lv;
}

// Traces the volume
void SingleScattering(CScene* pScene, CScene* pDevScene, unsigned int* pSeeds, CColorXyz* pDevEstFrameXyz)
{
	const dim3 KernelBlock(pScene->m_KernelSize.x, pScene->m_KernelSize.y);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));
	
	// Execute kernel
	KrnlSS<<<KernelGrid, KernelBlock>>>(pDevScene, pSeeds, pDevEstFrameXyz);
}
