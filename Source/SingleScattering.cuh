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

	float StepSize = 0.0005f;

	// Exit if beyond kernel boundaries
	if (X >= pScene->m_Camera.m_Film.m_Resolution.GetResX() || Y >= pScene->m_Camera.m_Film.m_Resolution.GetResY())
		return;
	
	// Init random number generator
	CCudaRNG RNG(&pSeeds[SID * 2], &pSeeds[SID * 2 + 1]);

	// Transmittance
	CColorXyz 	EyeTr	= SPEC_WHITE;		// Eye transmittance
	CColorXyz	L		= SPEC_BLACK;		// Measured volume radiance

	// Continue
	bool Continue = true;

	CRay EyeRay, RayCopy;

	float BoxMinT = 0.0f, BoxMaxT = 0.0f;

 	// Generate the camera ray
 	pScene->m_Camera.GenerateRay(Vec2f(X, Y), RNG.Get2(), EyeRay.m_O, EyeRay.m_D);

	EyeRay.m_MinT = 0.0f; 
	EyeRay.m_MaxT = FLT_MAX;

	CColorXyz Li = SPEC_BLACK;

	Vec3f Pe, Normal;
	
	if (SampleDistanceRM(EyeRay, RNG, Pe, pScene, 0))
	{
		RayCopy = CRay(EyeRay.m_O, EyeRay.m_D, 0.0f, (Pe - EyeRay.m_O).Length());

// 		if (NearestLight(pScene, RayCopy, Li))
// 		{
// 			pDevEstFrameXyz[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = Li;
// 			return;
// 		}

		// Fetch density
		const float D = Density(pScene, Pe);

		// Get opacity at eye point
		const float		Tr = GetOpacity(pScene, D).r;
		const CColorXyz	Ke = GetEmission(pScene, D).ToXYZ();
		
		// Compute outgoing direction
		const Vec3f Wo = Normalize(-EyeRay.m_D);

		// Obtain normal
		Normal = NormalizedGradient(pScene, Pe);

		// Estimate direct light at eye point
	 	L += EyeTr * UniformSampleOneLight(pScene, Wo, Pe, Normal, RNG, 0.001f);
	}

/*
	RayCopy.m_O		= Pe;
	RayCopy.m_D		= EyeRay.m_D;
	RayCopy.m_MinT	= EyeT;
	RayCopy.m_MaxT	= 10000000.0f;

	if (NearestLight(pScene, RayCopy, Li))
		Li += EyeTr * Li;
*/

	// Contribute
	pDevEstFrameXyz[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = L;
}

// Traces the volume
void SingleScattering(CScene* pScene, CScene* pDevScene, unsigned int* pSeeds, CColorXyz* pDevEstFrameXyz)
{
	const dim3 KernelBlock(16, 8);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));
	
	// Execute kernel
	KrnlSS<<<KernelGrid, KernelBlock>>>(pDevScene, pSeeds, pDevEstFrameXyz);
}
