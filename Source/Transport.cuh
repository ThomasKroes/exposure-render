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

#include "Shader.cuh"
#include "RayMarching.cuh"
#include "General.cuh"

//using namespace ExposureRender;

DEV void SampleLightSurface(LightSurfaceSample& LSS, CRNG& RNG, Vec3f P)
{
	const int LID = floorf(RNG.Get1() * gLights.m_NoLights);

	ErLight& L = gLights.m_LightList[LID];

	SurfaceSample SS;

	switch (L.m_Type)
	{
		case 0:
		{
			switch (L.m_ShapeType)
			{
				case 0:	SamplePlane(SS, RNG.Get2(), Vec2f(L.m_Size[0], L.m_Size[1]));		break;
				case 1:	SampleDisk(SS, RNG.Get2(), L.m_OuterRadius);						break;
				case 2:	SampleRing(SS, RNG.Get2(), L.m_InnerRadius, L.m_OuterRadius);		break;
				case 3:	SampleBox(SS, RNG.Get3(), ToVec3f(L.m_Size));						break;
				case 4:	SampleSphere(SS, RNG.Get2(), L.m_OuterRadius);						break;
			}

			break;
		}
		
		case 1:	SampleSphere(SS, RNG.Get2(), L.m_InnerRadius);	break;
	}

	LSS.P	= TransformPoint(L.m_TM, SS.P);
	LSS.Pdf	= DistanceSquared(P, LSS.P) / SS.Area;

	switch (L.m_Type)
	{
		// Area light
		case 0:
		{
			LSS.Le = ColorXYZf(L.m_Color[0], L.m_Color[1], L.m_Color[2]) / SS.Area;
			break;
		}

		// Environment light
		case 1:
		{
			switch (L.m_TextureType)
			{
				// Uniform color
				case 0:
				{
					LSS.Le = ColorXYZf(L.m_Color[0], L.m_Color[1], L.m_Color[2]) / SS.Area;
					break;
				}

				// Gradient
				case 1:
				{
					float4 Col = tex1D(gTexEnvironmentGradient, SS.UV[1]);
					LSS.Le = ColorXYZf(Col.x, Col.y, Col.z) / SS.Area;
					break;
				}
			}

			break;
		}
	}
}

DEV void HitTestLight(ErLight& Light, Ray R, RaySample& RS, bool RespectVisibility)
{
	if (RespectVisibility && !Light.m_Visible)
		return;

	// Transform ray into local shape coordinates
	Ray TR = TransformRay(R, Light.m_InvTM);

	// Result of intersection
	Intersection Int;

	switch (Light.m_Type)
	{
		// Intersect with area light
		case 0:
		{
			switch (Light.m_ShapeType)
			{
				case 0:
				{
					Int = IntersectPlane(TR, Light.m_OneSided, Vec2f(Light.m_Size[0], Light.m_Size[1]));
					break;
				}

				case 1:
				{
					Int = IntersectDisk(TR, Light.m_OneSided, Light.m_OuterRadius);
					break;
				}

				case 2:
				{
					Int = IntersectRing(TR, Light.m_OneSided, Light.m_InnerRadius, Light.m_OuterRadius);
					break;
				}

				case 3:
				{
					Int = IntersectBox(TR, ToVec3f(Light.m_Size), NULL);
					break;
				}

				case 4:
				{
					Int = IntersectSphere(TR, Light.m_OuterRadius);
					break;
				}

				case 5:
				{
					Int = IntersectCylinder(TR, Light.m_OuterRadius, Light.m_Size[1]);
					break;
				}
			}

			break;
		}

		case 1:
		{
			Int = IntersectSphere(TR, Light.m_InnerRadius);
			break;
		}
	}

	ColorXYZf Le;

	if (Int.Valid)
	{
		const Vec3f Pw = TransformPoint(Light.m_TM, Int.P);
		const Vec3f Nw = TransformVector(Light.m_TM, Int.N);
		const float Tw = Length(Pw - R.O);

		// Compute PDF
		const float Pdf = 1.0f / Light.m_Area;

		switch (Light.m_Type)
		{
			// Area light
			case 0:
			{
				Le = ColorXYZf(Light.m_Color[0], Light.m_Color[1], Light.m_Color[2]) / Light.m_Area;
				break;
			}

			// Environment light
			case 1:
			{
				switch (Light.m_TextureType)
				{
					// Uniform color
					case 0:
					{
						Le = ColorXYZf(Light.m_Color[0], Light.m_Color[1], Light.m_Color[2]) / Light.m_Area;
						break;
					}

					// Gradient
					case 1:
					{
						float4 Col = tex1D(gTexEnvironmentGradient, 0.5f + 0.5f * Normalize(Pw)[1]);
						Le = ColorXYZf(Col.x, Col.y, Col.z) / Light.m_Area;
						break;
					}
				}

				break;
			}
		}

		RS.SetValid(Tw, Pw, Nw, -R.D, Le, Int.UV, Pdf);
	}
}

DEV inline void SampleLights(Ray R, RaySample& RS, bool RespectVisibility = false)
{
	float T = FLT_MAX;

	for (int i = 0; i < gLights.m_NoLights; i++)
	{
		ErLight& Light = gLights.m_LightList[i];
		
		RaySample LocalRS(RaySample::Light);

		HitTestLight(Light, R, LocalRS, RespectVisibility);

		if (LocalRS.Valid && LocalRS.T < T)
		{
			RS = LocalRS;
			T = LocalRS.T;
		}
	}
}

DEV ColorXYZf SampleLight(CVolumeShader::EType Type, LightingSample& LS, RaySample RS, CRNG& RNG, CVolumeShader& Shader)
{
	Ray Rl;

	float ShaderPdf = 1.0f;

	Vec3f Wi;

	RaySample RSn(RaySample::Light);

	LightSurfaceSample LSS;

	SampleLightSurface(LSS, RNG, RS.P);
	
	Rl.O		= LSS.P;
	Rl.D		= Normalize(RS.P - LSS.P);
	Rl.MinT		= 0.0f;
	Rl.MaxT		= (RS.P - LSS.P).Length();

	Wi = -Rl.D; 

	ColorXYZf F = Shader.F(RS.Wo, Wi); 

	ShaderPdf = Shader.Pdf(RS.Wo, Wi);

	if (!LSS.Le.IsBlack() && ShaderPdf > 0.0f && LSS.Pdf > 0.0f && !FreePathRM(Rl, RNG))
	{
		const float WeightMIS = PowerHeuristic(1.0f, LSS.Pdf, 1.0f, ShaderPdf);
		
		if (Type == CVolumeShader::Brdf)
			return F * LSS.Le * AbsDot(Wi, RS.N) * WeightMIS / LSS.Pdf;

		if (Type == CVolumeShader::Phase)
			return F * LSS.Le * WeightMIS / LSS.Pdf;
	}

	return ColorXYZf(0.0f);
}

DEV ColorXYZf SampleShader(CVolumeShader::EType Type, LightingSample& LS, RaySample RS, CRNG& RNG, CVolumeShader& Shader)
{
	ColorXYZf Ld;

	Ray Rl;

	float ShaderPdf = 1.0f;

	Vec3f Wi;

	ColorXYZf F = Shader.SampleF(RS.Wo, Wi, ShaderPdf, LS.m_BsdfSample);

	if (!F.IsBlack() && ShaderPdf > 0.0f)
	{
		RaySample RSn(RaySample::Light);

		SampleLights(Ray(RS.P, Wi, 0.0f), RSn);

		if (RSn.Valid)
		{
			// Compose light ray
			Rl.O		= RSn.P;
			Rl.D		= Normalize(RS.P - RSn.P);
			Rl.MinT		= 0.0f;
			Rl.MaxT		= (RS.P - RSn.P).Length();

			if (RSn.Pdf > 0.0f && !RSn.Le.IsBlack() && !FreePathRM(Rl, RNG)) 
			{
				const float WeightMIS = PowerHeuristic(1.0f, ShaderPdf, 1.0f, RSn.Pdf);

				if (Type == CVolumeShader::Brdf)
					return F * RSn.Le * AbsDot(Wi, RSn.N) * WeightMIS / ShaderPdf;

				if (Type == CVolumeShader::Phase)
					return F * RSn.Le * WeightMIS / ShaderPdf;
			}
		}
	}

	return ColorXYZf(0.0f);
}

DEV ColorXYZf EstimateDirectLight(CVolumeShader::EType Type, LightingSample& LS, RaySample RS, CRNG& RNG, CVolumeShader& Shader)
{
	ColorXYZf Ld;
	
	if (gScattering.SamplingStrategy == 0 || gScattering.SamplingStrategy == 2)
		Ld += SampleLight(Type, LS, RS, RNG, Shader);

	if (gScattering.SamplingStrategy == 1 || gScattering.SamplingStrategy == 2)
		Ld += SampleShader(Type, LS, RS, RNG, Shader);

	return Ld;
}

DEV ColorXYZf UniformSampleOneLight(CVolumeShader::EType Type, RaySample RS, CRNG& RNG, CVolumeShader& Shader)
{
	LightingSample LS;

	LS.LargeStep(RNG);

	return EstimateDirectLight(Type, LS, RS, RNG, Shader);
}

DEV ColorXYZf UniformSampleOneLightVolume(RaySample RS, CRNG& RNG)
{
	const float I = GetIntensity(RS.P);

//	Lv += GetEmission(Intensity);

	switch (gVolume.m_ShadingType)
	{
		case 0:
		{
			CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), 5.0f, GetGlossiness(I));
			return UniformSampleOneLight(CVolumeShader::Brdf, RS, RNG, Shader);
		}
	
		case 1:
		{
			CVolumeShader Shader(CVolumeShader::Phase, RS.N, RS.Wo, GetDiffuse(I), GetSpecular(I), 5.0f, GetGlossiness(I));
			return UniformSampleOneLight(CVolumeShader::Phase, RS, RNG, Shader);
		}

		/*
		case 2:
		{
			const float GradMag = GradientMagnitude(Pe) * gVolume.m_IntensityInvRange;
			const float PdfBrdf = (1.0f - __expf(-pScene->m_GradientFactor * GradMag));

			if (RNG.Get1() < PdfBrdf)
				Lv += UniformSampleOneLight(pScene, CVolumeShader::Brdf, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe), RNG, true);
			else
				Lv += 0.5f * UniformSampleOneLight(pScene, CVolumeShader::Phase, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe), RNG, false);

			break;
		}
		*/
	}

	return ColorXYZf(0.0f);
}

DEV ColorXYZf UniformSampleOneLightReflector(RaySample RS, CRNG& RNG)
{
	CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, ColorXYZf(gReflectors.ReflectorList[RS.ReflectorID].DiffuseColor), ColorXYZf(gReflectors.ReflectorList[RS.ReflectorID].SpecularColor), gReflectors.ReflectorList[RS.ReflectorID].Ior, gReflectors.ReflectorList[RS.ReflectorID].Glossiness);
	//CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, ColorXYZf(0.5f), ColorXYZf(0.5f), 2.5f, 500.0f);
	return UniformSampleOneLight(CVolumeShader::Brdf, RS, RNG, Shader);
}














DEV void HitTestReflector(ErReflector& Reflector, Ray R, RaySample& RS)
{
	Ray TR = TransformRay(R, Reflector.InvTM);

	Intersection Int = IntersectPlane(TR, false, Vec2f(Reflector.Size[0], Reflector.Size[1]));

	if (Int.Valid)
	{
		const Vec3f Pw = TransformPoint(Reflector.TM, Int.P);
		const Vec3f Nw = TransformVector(Reflector.TM, Int.N);
		const float Tw = Length(Pw - R.O);

		RS.SetValid(Tw, Pw, Nw, -R.D, ColorXYZf(0.0f), Int.UV, 1.0f);
	}
}

DEV inline void SampleReflectors(Ray R, RaySample& RS)
{
	float T = FLT_MAX;

	for (int i = 0; i < gReflectors.NoReflectors; i++)
	{
		ErReflector& RO = gReflectors.ReflectorList[i];

		RaySample LocalRS(RaySample::Reflector);

		LocalRS.ReflectorID = i;

		HitTestReflector(RO, R, LocalRS);

		if (LocalRS.Valid && LocalRS.T < T)
		{
			RS = LocalRS;
			T = LocalRS.T;
		}
	}
}