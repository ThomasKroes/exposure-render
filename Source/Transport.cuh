#pragma once

#include "Geometry.h"
#include "Material.h"
#include "Scene.h"

#include "RayMarching.cuh"
#include "Woodcock.cuh"

DEV CColorXyz EstimateDirectLightBrdf(CScene* pScene, CLight& Light, CLightingSample& LS, const Vec3f& Wo, const Vec3f& Pe, const Vec3f& N, CCudaRNG& Rnd)
{
	CColorXyz Ld = SPEC_BLACK, Li = SPEC_BLACK, F = SPEC_BLACK;
	
	const float D = Density(pScene, Pe);

	CBSDF Bsdf(N, Wo, GetDiffuse(pScene, D).ToXYZ(), GetSpecular(pScene, D).ToXYZ(), pScene->m_IOR, GetRoughness(pScene, D).r);
	
	// Light/shadow ray
	CRay Rl; 

	// Light probability
	float LightPdf = 1.0f, BsdfPdf = 1.0f;
	
	// Incident light direction
	Vec3f Wi, P, Pl;

	// Sample the light source
 	Li = Light.SampleL(Pe, Rl, LightPdf, LS);
	
	CLight* pLight = NULL;

	Wi = -Rl.m_D; 

	F = Bsdf.F(Wo, Wi); 

	BsdfPdf	= Bsdf.Pdf(Wo, Wi);

	// Sample the light with MIS
	if (!Li.IsBlack() && BsdfPdf > 0.0f && LightPdf > 0.0f && !FreePathRM(Rl, Rnd, P, pScene))
	{
		const float WeightMIS = PowerHeuristic(1.0f, LightPdf, 1.0f, BsdfPdf);
 
		Ld += F * Li * (AbsDot(Wi, N) * WeightMIS / LightPdf);
	}

	// Sample the BRDF with MIS
	F = Bsdf.SampleF(Wo, Wi, BsdfPdf, LS.m_BsdfSample);
	
	if (!F.IsBlack() && BsdfPdf > 0.0f)
	{
		float WeightMIS = 1.0f;

		LightPdf = Light.Pdf(Pe, Wi);

		if (LightPdf == 0.0f)
			return Ld;
		
		WeightMIS = PowerHeuristic(1.0f, BsdfPdf, 1.0f, LightPdf);

		if (NearestLight(pScene, CRay(Pe, Wi, 0.0f), Li, Pl, pLight, &LightPdf) && pLight == &Light)
		{
			if (!Li.IsBlack() && !FreePathRM(CRay(Pl, Normalize(Pe - Pl), 0.0f, (Pe - Pl).Length()), Rnd, P, pScene)) 
			{
				Ld += F * Li * AbsDot(Wi, N) * WeightMIS / BsdfPdf;
			}
		}
	}

	return Ld;
}

DEV CColorXyz EstimateDirectLightPhase(CScene* pScene, CLight& Light, CLightingSample& LS, const Vec3f& Wo, const Vec3f& Pe, const Vec3f& N, CCudaRNG& Rnd)
{
	const float D = Density(pScene, Pe);

	// Accumulated radiance
	CColorXyz Ld = SPEC_BLACK, Li = SPEC_BLACK, F = INV_4_PI_F * GetDiffuse(pScene, D).ToXYZ();

	// Light/shadow ray
	CRay Rl; 

	// Light probability
	float LightPdf = 1.0f, PhasePdf = 1.0f;
	
	// Incident light direction
	Vec3f Wi, P, Pl;

	CLight* pLight = NULL;

	// Sample the light source
 	Li = Light.SampleL(Pe, Rl, LightPdf, LS);
	
	Wi = -Rl.m_D; 

	// Sample the light with MIS
	if (!Li.IsBlack() && LightPdf > 0.0f && !FreePathRM(Rl, Rnd, P, pScene))
	{
		const float WeightMIS = PowerHeuristic(1.0f, LightPdf, 1.0f, PhasePdf);

		Ld += F * Li * WeightMIS / LightPdf;
	}

	PhasePdf = INV_4_PI_F;
	
	// Sample the phase function with MIS
	Wi = UniformSampleSphere(LS.m_BsdfSample.m_Dir);

	if (!F.IsBlack() && PhasePdf > 0.0f)
	{
		float WeightMIS = 1.0f;

		LightPdf = Light.Pdf(Pe, Wi);

		if (LightPdf == 0.0f)
			return Ld;

		WeightMIS = PowerHeuristic(1.0f, PhasePdf, 1.0f, LightPdf);

		if (NearestLight(pScene, CRay(Pe, Wi, 0.0f), Li, Pl, pLight, &LightPdf) && pLight == &Light)
		{
			if (!Li.IsBlack() && !FreePathRM(CRay(Pl, Normalize(Pe - Pl), 0.0f, (Pe - Pl).Length()), Rnd, P, pScene)) 
			{
				Ld += F * Li * WeightMIS / PhasePdf;
			}
		}
	}

	return Ld;
}

DEV CColorXyz UniformSampleOneLight(CScene* pScene, const Vec3f& Wo, const Vec3f& Pe, const Vec3f& N, CCudaRNG& Rnd, const bool& Brdf)
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

	if (Brdf)
		return (float)NumLights * EstimateDirectLightBrdf(pScene, Light, LS, Wo, Pe, N, Rnd);
	else
		return (float)NumLights * EstimateDirectLightPhase(pScene, Light, LS, Wo, Pe, N, Rnd);
}