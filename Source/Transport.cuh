/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Light.cuh"
#include "Reflector.cuh"

DEV ColorXYZf SampleLight(CVolumeShader::EType Type, LightingSample& LS, RaySample RS, CRNG& RNG, CVolumeShader& Shader, int LightID)
{
	LightSurfaceSample LSS;

	SampleLightSurface(LSS, RNG, RS.P, LightID);

	if (LSS.Le.IsBlack() || LSS.Pdf <= 0.0f)
		return SPEC_BLACK;

	ColorXYZf F = Shader.F(RS.Wo, -LSS.Wi); 

	if (F.IsBlack())
		return SPEC_BLACK;

	float ShaderPdf = Shader.Pdf(RS.Wo, -LSS.Wi);

	if (ShaderPdf <= 0.0f)
		return SPEC_BLACK;

	if (!FreePathRM(Ray(LSS.P, LSS.Wi, 0.0f, (RS.P - LSS.P).Length()), RNG))
	{
		const float WeightMIS = PowerHeuristic(1.0f, LSS.Pdf, 1.0f, ShaderPdf);
		
		if (Type == CVolumeShader::Brdf)
			return F * LSS.Le * AbsDot(LSS.Wi, RS.N) * WeightMIS / LSS.Pdf;

		if (Type == CVolumeShader::Phase)
			return F * LSS.Le * WeightMIS / LSS.Pdf;
	}

	return SPEC_BLACK;
}

DEV ColorXYZf SampleShader(CVolumeShader::EType Type, LightingSample& LS, RaySample RS, CRNG& RNG, CVolumeShader& Shader, int LightID)
{
	ColorXYZf Ld;

	float ShaderPdf = 1.0f;

	Vec3f Wi;

	ColorXYZf F = Shader.SampleF(RS.Wo, Wi, ShaderPdf, LS.m_BsdfSample);

	if (F.IsBlack() || ShaderPdf <= 0.0f)
		return SPEC_BLACK;
	
	RaySample HitRS(RaySample::Light);

	IntersectAreaLights(Ray(RS.P, Wi), HitRS);

	if (!HitRS.Valid || HitRS.LightID != LightID || HitRS.Pdf <= 0.0f)
		return SPEC_BLACK;

	if (!FreePathRM(Ray(HitRS.P, Normalize(RS.P - HitRS.P), 0.0f, (RS.P - HitRS.P).Length()), RNG)) 
	{
		float Weight = PowerHeuristic(1.0f, ShaderPdf, 1.0f, HitRS.Pdf);

		if (Type == CVolumeShader::Brdf)
			return F * HitRS.Le * AbsDot(Normalize(HitRS.P - RS.P), RS.N) * Weight / ShaderPdf;

		if (Type == CVolumeShader::Phase)
			return F * HitRS.Le * Weight / ShaderPdf;
	}

	return SPEC_BLACK;
}

DEV ColorXYZf EstimateDirectLight(CVolumeShader::EType Type, LightingSample& LS, RaySample RS, CRNG& RNG, CVolumeShader& Shader, int LightID)
{
	ColorXYZf Ld;
	
	if (gScattering.SamplingStrategy == 0 || gScattering.SamplingStrategy == 2)
		Ld += SampleLight(Type, LS, RS, RNG, Shader, LightID);

	if (gScattering.SamplingStrategy == 1 || gScattering.SamplingStrategy == 2)
		Ld += SampleShader(Type, LS, RS, RNG, Shader, LightID);

	return (float)gLights.NoLights * Ld;
}

DEV ColorXYZf UniformSampleOneLight(CVolumeShader::EType Type, RaySample RS, CRNG& RNG, CVolumeShader& Shader)
{
	LightingSample LS;

	LS.LargeStep(RNG);

	const int LightID = floorf(RNG.Get1() * gLights.NoLights);

	return EstimateDirectLight(Type, LS, RS, RNG, Shader, LightID);
}

DEV ColorXYZf UniformSampleOneLightVolume(RaySample RS, CRNG& RNG)
{
	const float I = GetIntensity(RS.P);

	switch (gVolume.m_ShadingType)
	{
		case 0:
		{
			CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), 50.0f, GetGlossiness(I));
			return GetEmission(I) + UniformSampleOneLight(CVolumeShader::Brdf, RS, RNG, Shader);
		}
	
		case 1:
		{
			CVolumeShader Shader(CVolumeShader::Phase, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), 50.0f, GetGlossiness(I));
			return GetEmission(I) + UniformSampleOneLight(CVolumeShader::Phase, RS, RNG, Shader);
		}

		case 2:
		{
			const float NormalizedGradientMagnitude = GradientMagnitude(RS.P) * gVolume.m_GradientMagnitudeInvRange;
			const float PdfBrdf = GetOpacity(RS.P) * (1.0f - __expf(-gVolume.m_GradientFactor * NormalizedGradientMagnitude));

			if (RNG.Get1() < PdfBrdf)
			{
				CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), 50.0f, GetGlossiness(I));
				return GetEmission(I) + UniformSampleOneLight(CVolumeShader::Brdf, RS, RNG, Shader);
			}
			else
			{
				CVolumeShader Shader(CVolumeShader::Phase, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), 50.0f, GetGlossiness(I));
				return GetEmission(I) + 0.5f * UniformSampleOneLight(CVolumeShader::Phase, RS, RNG, Shader);
			}
		}

		case 3:
		{
			const float NormalizedGradientMagnitude = GradientMagnitude(RS.P) * gVolume.m_GradientMagnitudeInvRange;
			const float PdfBrdf = NormalizedGradientMagnitude;

			if (RNG.Get1() < PdfBrdf)
			{
				CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), 50.0f, GetGlossiness(I));
				return GetEmission(I) + UniformSampleOneLight(CVolumeShader::Brdf, RS, RNG, Shader);
			}
			else
			{
				CVolumeShader Shader(CVolumeShader::Phase, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), 50.0f, GetGlossiness(I));
				return GetEmission(I) + 0.5f * UniformSampleOneLight(CVolumeShader::Phase, RS, RNG, Shader);
			}
		}

		case 4:
		{
			const float NormalizedGradientMagnitude = GradientMagnitude(RS.P) * gVolume.m_GradientMagnitudeInvRange;

			return NormalizedGradientMagnitude > 0.0f && NormalizedGradientMagnitude <= 1.0f ? ColorXYZf(10.0f, 0.0f, 0.0f) : ColorXYZf(0.0f, 10.0f, 0.0f);

			if (NormalizedGradientMagnitude < gVolume.GradientThreshold)
			{
				CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), 50.0f, GetGlossiness(I));
				return GetEmission(I) + UniformSampleOneLight(CVolumeShader::Brdf, RS, RNG, Shader);
			}
			else
			{
				CVolumeShader Shader(CVolumeShader::Phase, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), 50.0f, GetGlossiness(I));
				return GetEmission(I) + 0.5f * UniformSampleOneLight(CVolumeShader::Phase, RS, RNG, Shader);
			}
		}
	}

	return ColorXYZf(0.0f);
}

DEV ColorXYZf UniformSampleOneLightReflector(RaySample RS, CRNG& RNG)
{
	CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, ColorXYZf(gReflectors.ReflectorList[RS.ReflectorID].DiffuseColor), ColorXYZf(gReflectors.ReflectorList[RS.ReflectorID].SpecularColor), gReflectors.ReflectorList[RS.ReflectorID].Ior, gReflectors.ReflectorList[RS.ReflectorID].Glossiness);
	
	return UniformSampleOneLight(CVolumeShader::Brdf, RS, RNG, Shader);
}