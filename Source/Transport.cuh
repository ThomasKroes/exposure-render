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

DEV ColorXYZf SampleLight(CRNG& RNG, const Vec3f& Pe, Vec3f& Pl, float& Pdf)
{
	const int LID = floorf(RNG.Get1() * gLights.m_NoLights);

	ErLight& L = gLights.m_LightList[LID];

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
					float4 Col = tex1D(gTexEnvironmentGradient, 0.5f + 0.5f * Normalize(Pl)[1]);
					Le = ColorXYZf(Col.x, Col.y, Col.z);
					break;
				}
			}

			break;
		}
	}

	return Le / L.m_Area;
}

DEV void HitTestLight(ErLight& Light, CRay R, RaySample& RS, bool RespectVisibility)
{
	if (RespectVisibility && !Light.m_Visible)
		return;

	// Transform ray into local shape coordinates
	CRay TR = TransformRay(R, Light.m_InvTM);

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
		const float Tw = Length(Pw - R.m_O);

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

		RS.SetValid(Tw, Pw, Nw, -R.m_D, Le, Int.UV, Pdf);
	}
}

DEV inline void SampleLights(CRay R, RaySample& RS, bool RespectVisibility = false)
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

	/*
	F = Shader.SampleF(RS.Wo, Wi, ShaderPdf, LS.m_BsdfSample);

	if (!F.IsBlack() && ShaderPdf > 0.0f)
	{
		RaySample RSn(RaySample::Light);

		SampleLights(CRay(RS.P, Wi, 0.0f), RSn);

		if (RSn.Valid)
		{
			// Compose light ray
			Rl.m_O		= Pl;
			Rl.m_D		= Normalize(RSn.P - Pl);
			Rl.m_MinT	= 0.0f;
			Rl.m_MaxT	= (RSn.P - Pl).Length();

			if (RSn.Pdf > 0.0f && !RSn.Le.IsBlack() && !FreePathRM(Rl, RNG)) 
			{
				const float WeightMIS = PowerHeuristic(1.0f, ShaderPdf, 1.0f, RSn.Pdf);

				if (Type == CVolumeShader::Brdf)
					Ld += F * Li * AbsDot(Wi, RSn.N) * WeightMIS / ShaderPdf;

				if (Type == CVolumeShader::Phase)
					Ld += F * Li * WeightMIS / ShaderPdf;
			}
		}
	}
	*/

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














DEV void HitTestReflector(ErReflector& Reflector, CRay R, RaySample& RS)
{
	CRay TR = TransformRay(R, Reflector.InvTM);

	Intersection Int = IntersectPlane(TR, false, Vec2f(Reflector.Size[0], Reflector.Size[1]));

	if (Int.Valid)
	{
		const Vec3f Pw = TransformPoint(Reflector.TM, Int.P);
		const Vec3f Nw = TransformVector(Reflector.TM, Int.N);
		const float Tw = Length(Pw - R.m_O);

		RS.SetValid(Tw, Pw, Nw, -R.m_D, ColorXYZf(0.0f), Int.UV, 1.0f);
	}
}

DEV inline void SampleReflectors(CRay R, RaySample& RS)
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