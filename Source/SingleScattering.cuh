#pragma once

#include "Transport.cuh"

#include <algorithm>
#include <vector>

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

	Vec3f Pe, Pl, Normal;
	
	CLight* pLight = NULL;

	int SpectralComponent = floorf(RNG.Get1() * 3.0f);

	if (SampleDistanceRM(Re, RNG, Pe, pScene, pScene->m_Spectral, SpectralComponent))
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, (Pe - Re.m_O).Length()), Li, Pl, pLight))
		{
			pDevEstFrameXyz[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X].c[SpectralComponent] = Li.c[SpectralComponent];
			return;
		}

		// Fetch density
		const float D = Density(pScene, Pe);

		// Get opacity at eye point
		const float		Tr = pScene->m_Spectral ? GetOpacity(pScene, D)[SpectralComponent] : GetOpacity(pScene, D).r;
		const CColorXyz	Ke = GetEmission(pScene, D).ToXYZ();
		
		Lv.c[SpectralComponent] += Ke.c[SpectralComponent];

		// Determine probabilities for picking brdf or phase function
		float PdfBrdf = pScene->m_TransferFunctions.m_Opacity.F(D).r * GradientMagnitude(pScene, Pe), PdfPhase = 1.0f - PdfBrdf;

		switch (pScene->m_ShadingType)
		{
			case 0:
			{
				// Brdf Shading
				Lv += UniformSampleOneLight(pScene, Normalize(-Re.m_D), Pe, NormalizedGradient(pScene, Pe), RNG, pScene->m_Spectral, SpectralComponent, true);
				break;
			}
		
			case 1:
			{
				// Brdf Shading
				Lv += 0.5f * UniformSampleOneLight(pScene, Normalize(-Re.m_D), Pe, NormalizedGradient(pScene, Pe), RNG, pScene->m_Spectral, SpectralComponent, false);
				break;
			}

			case 2:
			{
				if ((GradientMagnitude(pScene, Pe)) * Tr > 0.5f)//RNG.Get1() < PdfBrdf)
				{
					// Estimate direct light at eye point using BRDF shading
  					Lv += UniformSampleOneLight(pScene, Normalize(-Re.m_D), Pe, NormalizedGradient(pScene, Pe), RNG, pScene->m_Spectral, SpectralComponent, true);// / PdfBrdf;
				}
				else
				{
					// Estimate direct light at eye point using the phase function
  					Lv += 0.5f * UniformSampleOneLight(pScene, Normalize(-Re.m_D), Pe, NormalizedGradient(pScene, Pe), RNG, pScene->m_Spectral, SpectralComponent, false);// / PdfPhase;
				}

				break;
			}
		}
	}
	else
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, INF_MAX), Li, Pl, pLight))
			Lv.c[SpectralComponent] = Li.c[SpectralComponent];
	}

	pDevEstFrameXyz[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = SPEC_BLACK;

	// Contribute
	if (pScene->m_Spectral)
		pDevEstFrameXyz[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X].c[SpectralComponent] = Lv[SpectralComponent];
	else
		pDevEstFrameXyz[Y * (int)pScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = Lv;
}

void SingleScattering(CScene* pScene, CScene* pDevScene, unsigned int* pSeeds, CColorXyz* pDevEstFrameXyz)
{
	const dim3 KernelBlock(16, 8);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));
	
	// Execute kernel
	KrnlSS<<<KernelGrid, KernelBlock>>>(pDevScene, pSeeds, pDevEstFrameXyz);
}