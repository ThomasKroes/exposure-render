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

DEV ColorXYZf SampleLight(CRNG& RNG, const Vec3f& Pe, Vec3f& Pl, float& Pdf)
{
	const int LID = floorf(RNG.Get1() * gLighting.m_NoLights);

	_Light& L = gLighting.m_Lights[LID];

	// Sample point in light coordinates
	Vec3f LocalP, LocalN;

	switch (L.m_Type)
	{
		// Sample area light
		case 0:
		{
			// Shape types
			switch (L.m_ShapeType)
			{
				// Plane
				case 0:
				{
					LocalP = SamplePlane(RNG.Get2(), ToVec3f(L.m_Size), &LocalN);
					break;
				}

				// Disk
				case 1:
				{
					LocalP = SampleDisk(RNG.Get2(), L.m_OuterRadius, &LocalN);
					break;
				}

				// Ring
				case 2:
				{
					LocalP = SampleRing(RNG.Get2(), L.m_InnerRadius, L.m_OuterRadius, &LocalN);
					break;
				}

				// Box
				case 3:
				{
					LocalP = SampleBox(RNG.Get3(), ToVec3f(L.m_Size), &LocalN);
					break;
				}

				// Sphere
				case 4:
				{
					LocalP = SampleSphere(RNG.Get2(), L.m_OuterRadius, &LocalN);
					break;
				}
			}

			break;
		}
		
		// Sample background light
		case 1:
		{
			LocalP = L.m_InnerRadius * SampleUnitSphere(RNG.Get2());
			break;
		}
	}

	Pl	= TransformPoint(L.m_TM, LocalP);
	Pdf	= DistanceSquared(Pe, Pl) / L.m_Area;

	ColorXYZf Le;

	switch (L.m_Type)
	{
		// Area light
		case 0:
		{
			Le = ColorXYZf(L.m_Color[0], L.m_Color[1], L.m_Color[2]);
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
					Le = ColorXYZf(L.m_Color[0], L.m_Color[1], L.m_Color[2]);
					break;
				}

				// Gradient
				case 1:
				{
					float4 Col = tex1D(gTexEnvironmentGradient, 0.5f + 0.5f * Normalize(Pl).y);
					Le = ColorXYZf(Col.x, Col.y, Col.z);
					break;
				}
			}

			break;
		}
	}

	return Le / L.m_Area;
}

DEV void HitTestLight(_Light& Light, CRay R, RaySample& RS, bool RespectVisibility)
{
	if (RespectVisibility && !Light.m_Visible)
		return;

	// Transform ray into local shape coordinates
	CRay TR = TransformRay(R, Light.m_InvTM);

	// Result of intersection
	int Res = 0;

	// Hit distance in local coordinates
	float To = 0.0f;

	switch (Light.m_Type)
	{
		// Intersect with area light
		case 0:
		{
			switch (Light.m_ShapeType)
			{
				case 0:
				{
					Res = IntersectPlane(TR, Light.m_OneSided, ToVec3f(Light.m_Size), &To);
					break;
				}

				case 1:
				{
					Res = IntersectDisk(TR, Light.m_OneSided, Light.m_OuterRadius, &To);
					break;
				}

				case 2:
				{
					Res = IntersectRing(TR, Light.m_OneSided, Light.m_InnerRadius, Light.m_OuterRadius, &To);
					break;
				}

				case 3:
				{
					Res = IntersectBox(TR, ToVec3f(Light.m_Size), &To, NULL);
					break;
				}

				case 4:
				{
					Res = IntersectSphere(TR, Light.m_OuterRadius, &To);
					break;
				}
			}

			break;
		}

		case 1:
		{
			Res = IntersectSphere(TR, Light.m_InnerRadius, &To);
			break;
		}
	}

	ColorXYZf Le;

	if (Res > 0)
	{
		// Compute intersection point in world coordinates
		const Vec3f Pw = TransformPoint(Light.m_TM, TR(To));

		// Compute world distance to intersection
		const float Tw = Length(Pw - R.m_O);

		// Compute PDF
		const float Pdf = 1.0f / Light.m_Area;

		if (Res == 2)
		{
			Le = ColorXYZf(0.0f);
		}
		else
		{
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
							float4 Col = tex1D(gTexEnvironmentGradient, 0.5f + 0.5f * Normalize(Pw).y);
							Le = ColorXYZf(Col.x, Col.y, Col.z) / Light.m_Area;
							break;
						}
					}

					break;
				}
			}
		}

		RS.SetValid(Tw, Pw, Vec3f(0.0f, 1.0f, 0.0f), -R.m_D, Le, Vec2f(0.0f), Pdf);
	}
}

DEV inline void SampleLights(CRay R, RaySample& RS, bool RespectVisibility = false)
{
	float T = FLT_MAX;

	for (int i = 0; i < gLighting.m_NoLights; i++)
	{
		_Light& Light = gLighting.m_Lights[i];
		
		RaySample LocalRS(RaySample::Light);

		HitTestLight(Light, R, LocalRS, RespectVisibility);

		if (LocalRS.Valid && LocalRS.T < T)
		{
			RS = LocalRS;
			T = LocalRS.T;
		}
	}
}

DEV ColorXYZf EstimateDirectLight(CVolumeShader::EType Type, LightingSample& LS, RaySample RS, CRNG& RNG, CVolumeShader& Shader)
{
	ColorXYZf Ld = SPEC_BLACK, Li = SPEC_BLACK, F = SPEC_BLACK;
	
	CRay Rl; 

	float LightPdf = 1.0f, ShaderPdf = 1.0f;

	Vec3f Wi, Pl;

	Li = SampleLight(RNG, RS.P, Pl, LightPdf);
	
	Rl.m_O		= Pl;
	Rl.m_D		= Normalize(RS.P - Pl);
	Rl.m_MinT	= 0.0f;
	Rl.m_MaxT	= (RS.P - Pl).Length();

	Wi = -Rl.m_D; 

	F = Shader.F(RS.Wo, Wi); 

	ShaderPdf = Shader.Pdf(RS.Wo, Wi);
	
	if (!Li.IsBlack() && ShaderPdf > 0.0f && LightPdf > 0.0f && !FreePathRM(Rl, RNG))
	{
		const float WeightMIS = PowerHeuristic(1.0f, LightPdf, 1.0f, ShaderPdf);
		
		if (Type == CVolumeShader::Brdf)
			Ld += F * Li * AbsDot(Wi, RS.N) * WeightMIS / LightPdf;

		if (Type == CVolumeShader::Phase)
			Ld += F * Li * WeightMIS / LightPdf;
	}

	
	F = Shader.SampleF(RS.Wo, Wi, ShaderPdf, LS.m_BsdfSample);

	if (!F.IsBlack() && ShaderPdf > 0.0f)
	{
		SampleLights(CRay(RS.P, Wi, 0.0f), RS);

		if (RS.Valid)
		{
			// Compose light ray
			Rl.m_O		= Pl;
			Rl.m_D		= Normalize(RS.P - Pl);
			Rl.m_MinT	= 0.0f;
			Rl.m_MaxT	= (RS.P - Pl).Length();

			if (LightPdf > 0.0f && !Li.IsBlack() && !FreePathRM(Rl, RNG)) 
			{
				const float WeightMIS = PowerHeuristic(1.0f, ShaderPdf, 1.0f, LightPdf);

				if (Type == CVolumeShader::Brdf)
					Ld += F * Li * AbsDot(Wi, RS.N) * WeightMIS / ShaderPdf;

				if (Type == CVolumeShader::Phase)
					Ld += F * Li * WeightMIS / ShaderPdf;
			}
		}
	}
	/**/

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
	CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, ColorXYZf(gReflections.m_ReflectionObjects[RS.ReflectorID].DiffuseColor), ColorXYZf(gReflections.m_ReflectionObjects[RS.ReflectorID].SpecularColor), gReflections.m_ReflectionObjects[RS.ReflectorID].Ior, gReflections.m_ReflectionObjects[RS.ReflectorID].Glossiness);
	//CVolumeShader Shader(CVolumeShader::Brdf, RS.N, RS.Wo, ColorXYZf(0.5f), ColorXYZf(0.5f), 2.5f, 500.0f);
	return UniformSampleOneLight(CVolumeShader::Brdf, RS, RNG, Shader);
}














DEV void HitTestReflector(_ReflectionObject& Reflector, CRay R, RaySample& RS)
{
	// Transform ray into local shape coordinates
	CRay TR = TransformRay(R, Reflector.m_InvTM);

	// Result of intersection
	int Res = 0;

	float To = 0.0f;

	Vec2f UV;

	Res = IntersectPlane(TR, false, Vec3f(Reflector.Size[0], Reflector.Size[1], 0.0f), &To, &UV);

	if (Res > 0)
	{
		// Compute intersection point in world coordinates
		const Vec3f Pw = TransformPoint(Reflector.m_TM, TR(To));

		// Compute world distance to intersection
		const float Tw = Length(Pw - R.m_O);

		RS.SetValid(Tw, Pw, Vec3f(0.0f, 1.0f, 0.0f), -R.m_D, ColorXYZf(0.0f), UV, 1.0f);
	}
}

DEV inline void SampleReflectors(CRay R, RaySample& RS)
{
	float T = FLT_MAX;

	for (int i = 0; i < gReflections.m_NoReflectionObjects; i++)
	{
		_ReflectionObject& RO = gReflections.m_ReflectionObjects[i];

		RaySample LocalRS(RaySample::Reflector);

		LocalRS.ReflectorID = i;

		HitTestReflector(RO, R, LocalRS);

		if (LocalRS.Valid)
		{
			RS = LocalRS;
			T = LocalRS.T;
		}
	}
}