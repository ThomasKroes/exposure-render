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
#include "Woodcock.cuh"
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

DEV bool HitTestLight(int LightID, CRay& R, float& T, ColorXYZf& Le, bool RespectVisibility, Vec2f* pUV = NULL, float* pPdf = NULL)
{
	_Light& L = gLighting.m_Lights[LightID];

	if (RespectVisibility && !L.m_Visible)
		return false;

	// Transform ray into local shape coordinates
	CRay TR = TransformRay(R, L.m_InvTM);

	// Result of intersection
	int Res = 0;

	// Hit distance in local coordinates
	float Tl = 0.0f;

	switch (L.m_Type)
	{
		// Intersect with area light
		case 0:
		{
			switch (L.m_ShapeType)
			{
				// Plane
				case 0:
				{
					Res = IntersectPlane(TR, L.m_OneSided, ToVec3f(L.m_Size), &Tl, pUV);
					break;
				}

				// Disk
				case 1:
				{
					Res = IntersectDisk(TR, L.m_OneSided, L.m_OuterRadius, &Tl, pUV);
					break;
				}

				// Ring
				case 2:
				{
					Res = IntersectRing(TR, L.m_OneSided, L.m_InnerRadius, L.m_OuterRadius, &Tl, pUV);
					break;
				}

				// Box
				case 3:
				{
					Res = IntersectBox(TR, ToVec3f(L.m_Size), &Tl, NULL);
					break;
				}

				// Sphere
				case 4:
				{
					Res = IntersectSphere(TR, L.m_OuterRadius, &Tl);
					break;
				}
			}

			break;
		}

		// Intersect with background light
		case 1:
		{
			Res = IntersectSphere(TR, L.m_InnerRadius, &Tl);
			break;
		}
	}

	if (Res > 0)
	{
		// Compute intersection point in world coordinates
		const Vec3f P = TransformPoint(L.m_TM, TR(Tl));

		// Compute world distance to intersection
		T = Length(P - R.m_D);

		// Compute PDF
		if (pPdf)
			*pPdf = 1.0f / L.m_Area;

		if (Res == 2)
		{
			Le = ColorXYZf(0.0f);
		}
		else
		{
			switch (L.m_Type)
			{
				// Area light
				case 0:
				{
					Le = ColorXYZf(L.m_Color[0], L.m_Color[1], L.m_Color[2]) / L.m_Area;
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
							Le = ColorXYZf(L.m_Color[0], L.m_Color[1], L.m_Color[2]) / L.m_Area;
							break;
						}

						// Gradient
						case 1:
						{
							float4 Col = tex1D(gTexEnvironmentGradient, 0.5f + 0.5f * Normalize(P).y);
							Le = ColorXYZf(Col.x, Col.y, Col.z) / L.m_Area;
							break;
						}
					}

					break;
				}
			}
		}

		return true;
	}

	return false;
}

DEV inline bool NearestLight(CRay R, ColorXYZf& LightColor, Vec3f& Pl, bool RespectVisibility = false, float* pPdf = NULL)
{
	bool Hit = false;
	
	float T = 0.0f;

	float Pdf = 0.0f;

	for (int LID = 0; LID < gLighting.m_NoLights; LID++)
	{
		if (HitTestLight(LID, R, T, LightColor, RespectVisibility, NULL, &Pdf))
		{
			Pl			= R(T);
			Hit			= true;
			R.m_MaxT	= T;
		}
	}
	
	if (pPdf)
		*pPdf = Pdf;

	return Hit;
}

DEV ColorXYZf EstimateDirectLight(CVolumeShader::EType Type, float Intensity, LightingSample& LS, Vec3f Wo, Vec3f Pe, Vec3f N, CRNG& RNG)
{
	ColorXYZf Ld = SPEC_BLACK, Li = SPEC_BLACK, F = SPEC_BLACK;
	
	CVolumeShader Shader(Type, N, Wo, GetDiffuse(Intensity), GetSpecular(Intensity), 5.0f, GetGlossiness(Intensity));

	CRay Rl; 

	float LightPdf = 1.0f, ShaderPdf = 1.0f;

	Vec3f Wi, Pl;

 	Li = SampleLight(RNG, Pe, Pl, LightPdf);
	
	Rl.m_O		= Pl;
	Rl.m_D		= Normalize(Pe - Pl);
	Rl.m_MinT	= 0.0f;
	Rl.m_MaxT	= (Pe - Pl).Length();

	Wi = -Rl.m_D; 

	F = Shader.F(Wo, Wi); 

	ShaderPdf = Shader.Pdf(Wo, Wi);
	
	if (!Li.IsBlack() && ShaderPdf > 0.0f && LightPdf > 0.0f && !FreePathRM(Rl, RNG))
	{
		const float WeightMIS = PowerHeuristic(1.0f, LightPdf, 1.0f, ShaderPdf);
		
		if (Type == CVolumeShader::Brdf)
			Ld += F * Li * AbsDot(Wi, N) * WeightMIS / LightPdf;

		if (Type == CVolumeShader::Phase)
			Ld += F * Li * WeightMIS / LightPdf;
	}
	
	F = Shader.SampleF(Wo, Wi, ShaderPdf, LS.m_BsdfSample);

	if (!F.IsBlack() && ShaderPdf > 0.0f)
	{
		if (NearestLight(CRay(Pe, Wi, 0.0f), Li, Pl, &LightPdf))
		{
			// Compose light ray
			Rl.m_O		= Pl;
			Rl.m_D		= Normalize(Pe - Pl);
			Rl.m_MinT	= 0.0f;
			Rl.m_MaxT	= (Pe - Pl).Length();

			if (LightPdf > 0.0f && !Li.IsBlack() && !FreePathRM(Rl, RNG)) 
			{
				const float WeightMIS = PowerHeuristic(1.0f, ShaderPdf, 1.0f, LightPdf);

				if (Type == CVolumeShader::Brdf)
					Ld += F * Li * AbsDot(Wi, N) * WeightMIS / ShaderPdf;

				if (Type == CVolumeShader::Phase)
					Ld += F * Li * WeightMIS / ShaderPdf;
			}
		}
	}
	/**/

	return Ld;
}

DEV ColorXYZf UniformSampleOneLight(CVolumeShader::EType Type, float Intensity, Vec3f Wo, Vec3f Pe, Vec3f N, CRNG& RNG)
{
	LightingSample LS;

	LS.LargeStep(RNG);

	return EstimateDirectLight(Type, Intensity, LS, Wo, Pe, N, RNG);
}