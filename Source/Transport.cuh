#pragma once

#include "Geometry.h"
#include "Material.h"
#include "Scene.h"

DEV inline CColorXyz Transmittance(CScene* pScene, const Vec3f& P, const Vec3f& D, const float& MaxT, const float& StepSize, CCudaRNG& Rnd)
{
	// Near and far intersections with volume axis aligned bounding box
	float NearT = 0.0f, FarT = 0.0f;

	// Intersect with volume axis aligned bounding box
	if (!pScene->m_BoundingBox.Intersect(CRay(P, D, 0.0f, FLT_MAX), &NearT, &FarT))
		return SPEC_BLACK;

	// Clamp to near plane if necessary
	if (NearT < 0.0f) 
		NearT = 0.0f;     

	CColorXyz Lt = SPEC_WHITE;

	NearT += Rnd.Get1() * StepSize;

	// Accumulate
	while (NearT < MaxT)
	{
		// Determine sample point
		const Vec3f SP = P + D * (NearT);

		// Fetch density
		const float D = Density(pScene, SP);
		
		// We ignore air density
		if (D == 0)
		{
			// Increase extent
			NearT += StepSize;
			continue;
		}

		// Get shadow opacity
		const float	Opacity = GetOpacity(pScene, D).r;

		if (Opacity > 0.0f)
		{
			// Compute eye transmittance
			Lt *= expf(-(Opacity * StepSize));

			// Exit if eye transmittance is very small
			if (Lt.y() < 0.05f)
				break;
		}

		// Increase extent
		NearT += StepSize;
	}

	return Lt;
}

DEV CColorXyz EstimateDirectLight(CScene* pScene, CLight& Light, CLightingSample& LS, const Vec3f& Wo, const Vec3f& Pe, const Vec3f& N, CCudaRNG& Rnd, const float& StepSize)
{
// 	if (Dot(Wo, N) < 0.0f)
// 		return SPEC_BLACK;

	// Accumulated radiance
	CColorXyz Ld = SPEC_BLACK;
	
	// Radiance from light source
	CColorXyz Li = SPEC_BLACK;

	// Attenuation
	CColorXyz Tr = SPEC_BLACK;

	const float D = Density(pScene, Pe);

	CBSDF Bsdf(N, Wo, GetDiffuse(pScene, D).ToXYZ(), GetSpecular(pScene, D).ToXYZ(), 50.0f, 0.0001 * GetRoughness(pScene, D).r);
	// Light/shadow ray
	CRay R; 

	// Light probability
	float LightPdf = 1.0f, BsdfPdf = 1.0f;
	
	// Incident light direction
	Vec3f Wi;

	CColorXyz F = SPEC_BLACK;
	
	CSurfacePoint SPe, SPl;

	SPe.m_P		= Pe;
	SPe.m_Ng	= N; 

	// Sample the light source
 	Li = Light.SampleL(Pe, R, LightPdf, LS);
	
	Wi = -R.m_D; 

	F = Bsdf.F(Wo, Wi); 

	BsdfPdf	= Bsdf.Pdf(Wo, Wi);
	
	// Sample the light with MIS
	if (!Li.IsBlack() && LightPdf > 0.0f && BsdfPdf > 0.0f)
	{
		// Compute tau
		const CColorXyz Tr = Transmittance(pScene, R.m_O, R.m_D, Length(R.m_O - Pe), StepSize, Rnd);
		
		// Attenuation due to volume
		Li *= Tr;

		// Compute MIS weight
		const float Weight = 1.0f;//PowerHeuristic(1.0f, LightPdf, 1.0f, BsdfPdf);
 
		// Add contribution
		Ld += F * Li * (AbsDot(Wi, N) * Weight / LightPdf);
	}

	return Ld;

	/*
	// Sample the BRDF with MIS
	F = Bsdf.SampleF(Wo, Wi, BsdfPdf, LS.m_BsdfSample);
	
	CLight* pNearestLight = NULL;

	Vec2f UV;
	
	if (!F.IsBlack())
	{
		float MaxT = 1000000000.0f; 

		// Compute virtual light point
		const Vec3f Pl = Pe + (MaxT * Wi);

		if (NearestLight(pScene, Pe, Wi, 0.0f, MaxT, pNearestLight, NULL, &UV, &LightPdf))
		{
			if (LightPdf > 0.0f && BsdfPdf > 0.0f) 
			{
				// Add light contribution from BSDF sampling
				const float Weight = PowerHeuristic(1.0f, BsdfPdf, 1.0f, LightPdf);
				 
				// Get exitant radiance from light source
// 				Li = pNearestLight->Le(UV, pScene->m_Materials, pScene->m_Textures, pScene->m_Bitmaps);

				if (!Li.IsBlack())
				{
					// Scale incident radiance by attenuation through volume
					Tr = Transmittance(pScene, Pe, Wi, 1.0f, StepSize, Rnd);

					// Attenuation due to volume
					Li *= Tr;

					// Contribute
					Ld += F * Li * AbsDot(Wi, N) * Weight / BsdfPdf;
				}
			}
		}
	}
	*/

	return Ld;
}

DEV CColorXyz UniformSampleOneLight(CScene* pScene, const Vec3f& Wo, const Vec3f& Pe, const Vec3f& N, CCudaRNG& Rnd, const float& StepSize)
{
	// Determine no. lights
	const int NumLights = pScene->m_Lighting.m_NoLights;

	// Exit return zero radiance if no light
 	if (NumLights == 0)
 		return SPEC_BLACK;

	CLightingSample LS;

	// Create light sampler
	LS.LargeStep(Rnd);

	// Choose which light to sample
	const int WhichLight = (int)floorf(LS.m_LightNum * (float)NumLights);

	// Get the light
	CLight& Light = pScene->m_Lighting.m_Lights[WhichLight];

	// Return estimated direct light
	return (float)NumLights * EstimateDirectLight(pScene, Light, LS, Wo, Pe, N, Rnd, StepSize);
}