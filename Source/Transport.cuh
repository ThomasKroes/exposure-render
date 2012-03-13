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

DEVICE_NI bool Intersect(Ray R, CRNG& RNG)
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

DEVICE_NI bool Visible(Vec3f P1, Vec3f P2, CRNG& RNG)
{
	if (!gScattering.Shadows)
		return true;

	Vec3f W = Normalize(P2 - P1);

	Ray R(P1 + W * EPS1, W, 0.0f, min((P2 - P1).Length() - EPS2, gScattering.MaxShadowDistance));

	return !Intersect(R, RNG);
}

DEVICE_NI ColorXYZf EstimateDirectLight(LightingSample& LS, ScatterEvent& SE, CRNG& RNG, CVolumeShader& Shader, int LightID)
{
	ErLight& Light = gLights.LightList[LightID];

	Vec3f Wi;

	// Incident radiance (Li), direct light (Ld)
	ColorXYZf Li = SPEC_BLACK, Ld = SPEC_BLACK;

	SurfaceSample SS;

	SampleLight(Light, LS.LightSample, SS, SE, Wi, Li);

	ColorXYZf F = Shader.F(SE.Wo, Wi);
	
	float BsdfPdf = Shader.Pdf(SE.Wo, Wi);

	if (!Li.IsBlack() && !F.IsBlack() && BsdfPdf > 0.0f && Visible(SE.P, SS.P, RNG))
	{
		const float LightPdf = DistanceSquared(SE.P, SS.P) / (AbsDot(SS.N, -Wi) * Light.Shape.Area);

		const float Weight = PowerHeuristic(1, LightPdf, 1, BsdfPdf);

		Ld += F * Li * (AbsDot(Wi, SE.N) * Weight / LightPdf);
	}

	F = Shader.SampleF(SE.Wo, Wi, BsdfPdf, LS.BrdfSample);

	if (F.IsBlack() || BsdfPdf <= 0.0f)
		return Ld;
	
	ScatterEvent SE2(ScatterEvent::Light);

	IntersectLights(Ray(SE.P, Wi), SE2);

	if (!SE2.Valid || SE2.LightID != LightID)
		return Ld;

	Li = SE2.Le;

	if (!Li.IsBlack() && Visible(SE.P, SE2.P, RNG))
	{
		Wi = Normalize(SE2.P - SE.P);

		const float LightPdf = DistanceSquared(SE.P, SE2.P) / (AbsDot(SE.N, -Wi) * Light.Shape.Area);

		const float Weight = PowerHeuristic(1, BsdfPdf, 1, LightPdf);

		Ld += F * Li * (AbsDot(Wi, SE.N) * Weight / BsdfPdf);
	}
	
	return Ld;
}

DEVICE_NI ColorXYZf UniformSampleOneLight(ScatterEvent& SE, CRNG& RNG, CVolumeShader& Shader, LightingSample& LS)
{
	const int LightID = floorf(LS.LightNum * gLights.NoLights);

	return (float)gLights.NoLights * EstimateDirectLight(LS, SE, RNG, Shader, LightID);
}

DEVICE_NI ColorXYZf UniformSampleOneLightVolume(ScatterEvent& SE, CRNG& RNG, LightingSample& LS)
{
	const float I = GetIntensity(SE.P);

	float PdfBrdf = 1.0f;

	switch (gVolume.ShadingType)
	{
		case 0:
		{
			PdfBrdf = 1.0f;
			break;
		}
	
		case 1:
		{
			PdfBrdf = 0.0f;
			break;
		}

		case 2:
		{
			const float NGM			= GradientMagnitude(SE.P) * gVolume.GradientMagnitudeRange.Inv;
			const float Sensitivity	= 25;
			const float ExpGF		= 3;
			const float Exponent	= Sensitivity * powf(gVolume.GradientFactor, ExpGF) * NGM;
			
			PdfBrdf = gScattering.OpacityModulated ? GetOpacity(SE.P) * (1.0f - __expf(-Exponent)) : 1.0f - __expf(-Exponent);
			break;
		}

		case 3:
		{
			const float NGM = GradientMagnitude(SE.P) * gVolume.GradientMagnitudeRange.Inv;
			
			PdfBrdf = 1.0f - powf(1.0f - NGM, 2.0f);
			break;
		}

		case 4:
		{
			const float NGM = GradientMagnitude(SE.P) * gVolume.GradientMagnitudeRange.Inv;

			if (NGM > gVolume.GradientThreshold)
			{
				CVolumeShader Shader(CVolumeShader::Brdf, SE.N, SE.Wo, GetDiffuse(I), GetSpecular(I), gScattering.IndexOfReflection, GetGlossiness(I));
				return GetEmission(I) + UniformSampleOneLight(SE, RNG, Shader, LS);
			}
			else
			{
				CVolumeShader Shader(CVolumeShader::Phase, SE.N, SE.Wo, GetDiffuse(I), GetSpecular(I), gScattering.IndexOfReflection, GetGlossiness(I));
				return GetEmission(I) + 0.5f * UniformSampleOneLight(SE, RNG, Shader, LS);
			}
		}
	}

	if (RNG.Get1() < PdfBrdf)
	{
		CVolumeShader Shader(CVolumeShader::Brdf, SE.N, SE.Wo, GetDiffuse(I), GetSpecular(I), gScattering.IndexOfReflection, GetGlossiness(I));
		return GetEmission(I) + UniformSampleOneLight(SE, RNG, Shader, LS);
	}
	else
	{
		CVolumeShader Shader(CVolumeShader::Phase, SE.N, SE.Wo, GetDiffuse(I), GetSpecular(I), gScattering.IndexOfReflection, GetGlossiness(I));
		return GetEmission(I) + UniformSampleOneLight(SE, RNG, Shader, LS) * 0.5f;
	}
}