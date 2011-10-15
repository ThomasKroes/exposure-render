#pragma once

#include "Shader.cuh"
#include "RayMarching.cuh"

DEV CColorXyz EstimateDirectLightBrdf(CScene* pScene, const float& Density, CLight& Light, CLightingSample& LS, const Vec3f& Wo, const Vec3f& Pe, const Vec3f& N, CRNG& Rnd)
{
	CColorXyz Ld = SPEC_BLACK, Li = SPEC_BLACK, F = SPEC_BLACK;
	
	CBSDF Bsdf(N, Wo, GetDiffuse(Density).ToXYZ(), GetSpecular(Density).ToXYZ(), 5.5f/*pScene->m_IOR*/, GetRoughness(Density));
	
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
	if (!Li.IsBlack() && BsdfPdf > 0.0f && LightPdf > 0.0f && !FreePathRM(Rl, Rnd))
	{
		const float WeightMIS = PowerHeuristic(1.0f, LightPdf, 1.0f, BsdfPdf);
 
		Ld += F * Li * (AbsDot(Wi, N) * WeightMIS / LightPdf);
	}

	/*
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
			if (!Li.IsBlack() && !FreePathRM(CRay(Pl, Normalize(Pe - Pl), 0.0f, (Pe - Pl).Length()), Rnd)) 
			{
				Ld += F * Li * AbsDot(Wi, N) * WeightMIS / BsdfPdf;
			}
		}
	}
	*/

	return Ld;
}

DEV CColorXyz EstimateDirectLightPhase(CScene* pScene, const float& Density, CLight& Light, CLightingSample& LS, const Vec3f& Wo, const Vec3f& Pe, const Vec3f& N, CRNG& Rnd)
{
	// Accumulated radiance
	CColorXyz Ld = SPEC_BLACK, Li = SPEC_BLACK, F = INV_4_PI_F * GetDiffuse(Density).ToXYZ();

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
	if (!Li.IsBlack() && LightPdf > 0.0f && !FreePathRM(Rl, Rnd))
	{
		const float WeightMIS = PowerHeuristic(1.0f, LightPdf, 1.0f, PhasePdf);

		Ld += F * Li * WeightMIS / LightPdf;
	}
	
	/*
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
			if (!Li.IsBlack() && !FreePathRM(CRay(Pl, Normalize(Pe - Pl), 0.0f, (Pe - Pl).Length()), Rnd)) 
			{
				Ld += F * Li * WeightMIS / PhasePdf;
			}
		}
	}
	*/

	return Ld;
}

DEV CColorXyz UniformSampleOneLight(CScene* pScene, const float& Density, const Vec3f& Wo, const Vec3f& Pe, const Vec3f& N, CRNG& Rnd, const bool& Brdf)
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
		return (float)NumLights * EstimateDirectLightBrdf(pScene, Density, Light, LS, Wo, Pe, N, Rnd);
	else
		return (float)NumLights * EstimateDirectLightPhase(pScene, Density, Light, LS, Wo, Pe, N, Rnd);
}