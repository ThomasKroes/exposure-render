#pragma once

#include "Geometry.h"
#include "curand_kernel.h"

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
KERNEL void KrnlSS(CScene* pScene, curandStateXORWOW_t* pDevRandomStates, CColorXyz* pDevEstFrameXyz)
{
	const int X = (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global y
	const int Y	= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global x
	
	// Compute sample ID
	const int SID = (Y * (gridDim.x * blockDim.x)) + X;

	float StepSize = 0.03;

	// Exit if beyond kernel boundaries
	if (X >= pScene->m_Camera.m_Film.m_Resolution.GetResX() || Y >= pScene->m_Camera.m_Film.m_Resolution.GetResY())
		return;
	
	// Init random number generator
	CCudaRNG RNG(&pDevRandomStates[SID]);

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

	// Check if ray passes through volume, if it doesn't, evaluate scene lights and stop tracing 
 	if (!NearestIntersection(pScene, EyeRay, StepSize, RNG.Get1(), &BoxMinT, &BoxMaxT))
 		Continue = false;

	CColorXyz Li = SPEC_BLACK;
	RayCopy = CRay(EyeRay.m_O, EyeRay.m_D, 0.0f, Continue ? EyeRay.m_MinT : EyeRay.m_MaxT);

	if (NearestLight(pScene, RayCopy, Li))
	{
		pDevEstFrameXyz[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = Li;
		return;
	}

	if (EyeRay.m_MaxT == INF_MAX)
 		Continue = false;
	
	float EyeT	= EyeRay.m_MinT;

	Vec3f EyeP, Normal;
	
	// Walk along the eye ray with ray marching
	while (Continue && EyeT < EyeRay.m_MaxT)
	{
		// Determine new point on eye ray
		EyeP = EyeRay(EyeT);

		// Increase parametric range
		EyeT += StepSize;

		// Fetch density
		const float D = Density(pScene, EyeP);

		// We ignore air density
		if (Density == 0) 
			continue;
		 
		// Get opacity at eye point
		const float		Tr = GetOpacity(pScene, D).r;
		const CColorXyz	Ke = GetEmission(pScene, D).ToXYZ();
		
		// Add emission
		EyeTr += Ke; 
		
		// Compute outgoing direction
		const Vec3f Wo = Normalize(-EyeRay.m_D);

		// Obtain normal
		Normal = NormalizedGradient(pScene, EyeP);//ComputeGradient(pScene, EyeP, Wo);

		// Exit if air, or not within hemisphere
		if (Tr < 0.05f)// || Dot(Wo, Normal[TID]) < 0.0f)
			continue;

		// Estimate direct light at eye point
	 	L += EyeTr * UniformSampleOneLight(pScene, Wo, EyeP, Normal, RNG, StepSize);

		// Compute eye transmittance
		EyeTr *= expf(-(Tr * StepSize));

		/*
		// Russian roulette
		if (EyeTr.y() < 0.5f)
		{
			const float DieP = 1.0f - (EyeTr.y() / Threshold);

			if (DieP > RNG.Get1())
			{
				break;
			}
			else
			{
				EyeTr *= 1.0f / (1.0f - DieP);
			}
		}
		*/

		if (EyeTr.y() < 0.05f)
			break;
	}

	RayCopy.m_O		= EyeP;
	RayCopy.m_D		= EyeRay.m_D;
	RayCopy.m_MinT	= EyeT;
	RayCopy.m_MaxT	= 10000000.0f;

	if (NearestLight(pScene, RayCopy, Li))
		Li += EyeTr * Li;

	// Contribute
	pDevEstFrameXyz[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = L;
}

// Traces the volume
void SingleScattering(CScene* pScene, CScene* pDevScene, curandStateXORWOW_t* pDevRandomStates, CColorXyz* pDevEstFrameXyz)
{
	// Copy the scene from host memory to device memory
//	cudaMemcpyToSymbol("gScene", pScene, sizeof(CScene));

	const dim3 KernelBlock(32, 8);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));
	
	// Execute kernel
	KrnlSS<<<KernelGrid, KernelBlock>>>(pDevScene, pDevRandomStates, pDevEstFrameXyz);
}
