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

#define EPS1 0.0001f
#define EPS2 0.0002f

DEV inline bool Intersect(Ray R, CRNG& RNG)
{
	ScatterEvent SE(ScatterEvent::Light);

	if (IntersectsLight(R))
		return true;

	if (IntersectsReflector(R))
		return true;

	if (ScatterEventInVolume(R, RNG))
		return true;

	return false;
}

DEV bool Visible(Vec3f P1, Vec3f P2, CRNG& RNG)
{
	Vec3f W = Normalize(P2 - P1);

	Ray R(P1 + W * EPS1, W, 0.0f, (P2 - P1).Length() - EPS2);

	return !Intersect(R, RNG);
}

DEV ColorXYZf EstimateDirectLight(CVolumeShader::EType Type, LightingSample& LS, ScatterEvent& SE, CRNG& RNG, CVolumeShader& Shader, int LightID)
{
	// Get light
	ErLight& Light = gLights.LightList[LightID];

	//float LightPdf; 
	Vec3f Wi;

	// Incident radiance (Li), direct light (Ld)
	ColorXYZf Li = SPEC_BLACK, Ld = SPEC_BLACK;

	SampleLight(Light, LS.m_LightSample, SE, Wi, Li);

	ColorXYZf F = Shader.F(SE.Wo, Wi); 
	
	float Ps = Shader.Pdf(SE.Wo, Wi);

	if (Li.IsBlack() || F.IsBlack() || Ps <= 0.0f)
		return Ld;

	if (Visible(LS.m_LightSample.SS.P, SE.P, RNG))
	{
		const float d2 = DistanceSquared(SE.P, LS.m_LightSample.SS.P);

		float Pl = 1.0f / Light.Shape.Area;

		Li *= F;
		Li /= d2;

		//if (mis)
		//{
		Li *= PowerHeuristic(1, Pl * d2 / AbsDot(Wi, SE.N), 1, Ps);
		//}

		// Add to direct light
		Ld += Li;
/*
//		const float WeightMIS = PowerHeuristic(1.0f, LightPdf, 1.0f, ShaderPdf);
		const float Pmz = 1.0f / Light.Shape.Area;

		float g = G(SE.P, SE.N, LS.m_LightSample.SS.P, LS.m_LightSample.SS.N);
		
		const float We = powf(Pmz, 2.0f) / (powf(Pmz, 2.0f) + powf(Pf * AbsDot(Wi, SE.N) * g, 2.0f));

		return We * F * AbsDot(Wi, SE.N) * g * (Le / Light.Shape.Area);

		
		if (Type == CVolumeShader::Brdf)
			return F * Le * AbsDot(Wi, SE.N) / LightPdf;
		
		if (Type == CVolumeShader::Phase)
			return F * Le * WeightMIS / LightPdf;
		*/
	}

	// Sample new ray direction
	F = Shader.SampleF(SE.Wo, Wi, Ps, LS.m_BsdfSample);

	if (F.IsBlack() || Ps <= 0.0f)
		return Ld;
	
	ScatterEvent SE2(ScatterEvent::Light);

	IntersectLights(Ray(SE.P, Wi), SE2);

	if (!SE2.Valid)
		return Ld;

	if (Visible(LS.m_LightSample.SS.P, SE.P, RNG))
	{
		Li = SE2.Le;

		float Pl = 1.0f / gLights.LightList[SE2.LightID].Shape.Area;

		// Compute distance squared
		const float d2 = DistanceSquared(SE.P, SE2.P);
		
		const float lightPdf2 = Pl * d2 / AbsDot(Wi, SE.N);

		const float weight = PowerHeuristic(1, Ps, 1, lightPdf2);

//		Ld += Li * weight;
	}

	return Ld;
}

DEV ColorXYZf UniformSampleOneLight(CVolumeShader::EType Type, ScatterEvent RS, CRNG& RNG, CVolumeShader& Shader)
{
	LightingSample LS;

	LS.LargeStep(RNG);

	const int LightID = floorf(RNG.Get1() * gLights.NoLights);

	return (float)gLights.NoLights * EstimateDirectLight(Type, LS, RS, RNG, Shader, LightID);
}

DEV ColorXYZf UniformSampleOneLightVolume(ScatterEvent RS, CRNG& RNG)
{
	const float I = GetIntensity(RS.P);

	switch (gVolume.ShadingType)
	{
		case 0:
		{
			CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), gScattering.IndexOfReflection, GetGlossiness(I));
			return GetEmission(I) + UniformSampleOneLight(CVolumeShader::Brdf, RS, RNG, Shader);
		}
	
		case 1:
		{
			CVolumeShader Shader(CVolumeShader::Phase, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), gScattering.IndexOfReflection, GetGlossiness(I));
			return GetEmission(I) + UniformSampleOneLight(CVolumeShader::Phase, RS, RNG, Shader);
		}

		case 2:
		{
			const float NGM = GradientMagnitude(RS.P) * gVolume.GradientMagnitudeRange.Inv;

			const float Sensitivity = 25;
			const float ExpGF = 3;

			const float Exponent = Sensitivity * powf(gVolume.GradientFactor, ExpGF) * NGM;

			const float PdfBrdf = gScattering.OpacityModulated ? GetOpacity(RS.P) * (1.0f - __expf(-Exponent)) : 1.0f - __expf(-Exponent);

			if (RNG.Get1() < PdfBrdf)
			{
				CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), gScattering.IndexOfReflection, GetGlossiness(I));
				return GetEmission(I) + UniformSampleOneLight(CVolumeShader::Brdf, RS, RNG, Shader);
			}
			else
			{
				CVolumeShader Shader(CVolumeShader::Phase, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), gScattering.IndexOfReflection, GetGlossiness(I));
				return GetEmission(I) + 0.5f * UniformSampleOneLight(CVolumeShader::Phase, RS, RNG, Shader);
			}
		}

		case 3:
		{
			const float NGM = GradientMagnitude(RS.P) * gVolume.GradientMagnitudeRange.Inv;

			const float PdfBrdf = 1.0f - powf(1.0f - NGM, 2.0f);

			if (RNG.Get1() < PdfBrdf)
			{
				CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), gScattering.IndexOfReflection, GetGlossiness(I));
				return GetEmission(I) + UniformSampleOneLight(CVolumeShader::Brdf, RS, RNG, Shader);
			}
			else
			{
				CVolumeShader Shader(CVolumeShader::Phase, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), gScattering.IndexOfReflection, GetGlossiness(I));
				return GetEmission(I) + 0.5f * UniformSampleOneLight(CVolumeShader::Phase, RS, RNG, Shader);
			}
		}

		case 4:
		{
			const float NormalizedGradientMagnitude = GradientMagnitude(RS.P) * gVolume.GradientMagnitudeRange.Inv;

			// return ColorXYZf(NormalizedGradientMagnitude);
			// return NormalizedGradientMagnitude > 0.0f && NormalizedGradientMagnitude <= 1.0f ? ColorXYZf(10.0f, 0.0f, 0.0f) : ColorXYZf(0.0f, 10.0f, 0.0f);

			if (NormalizedGradientMagnitude > gVolume.GradientThreshold)
			{
				CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), gScattering.IndexOfReflection, GetGlossiness(I));
				return GetEmission(I) + UniformSampleOneLight(CVolumeShader::Brdf, RS, RNG, Shader);
			}
			else
			{
				CVolumeShader Shader(CVolumeShader::Phase, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), gScattering.IndexOfReflection, GetGlossiness(I));
				return GetEmission(I) + 0.5f * UniformSampleOneLight(CVolumeShader::Phase, RS, RNG, Shader);
			}
		}
	}

	return ColorXYZf(0.0f);
}

DEV ColorXYZf UniformSampleOneLightReflector(ScatterEvent RS, CRNG& RNG)
{
	CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, ColorXYZf(gReflectors.ReflectorList[RS.ReflectorID].DiffuseColor), ColorXYZf(gReflectors.ReflectorList[RS.ReflectorID].SpecularColor), gReflectors.ReflectorList[RS.ReflectorID].Ior, gReflectors.ReflectorList[RS.ReflectorID].Glossiness);
	
	return UniformSampleOneLight(CVolumeShader::Brdf, RS, RNG, Shader);
}