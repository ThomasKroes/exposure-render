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

	switch (gLighting.m_Type[LID])
	{
		// Sample area light
		case 0:
		{
			switch (gLighting.m_ShapeType[LID])
			{
				// Plane
				case 0:
				{
					/*
					Pl	= ToVec3f(gLighting.m_P[LID]);
					Pdf	= (Pe - Pl).Length();

					return ColorXYZf(gLighting.m_Color[LID].x, gLighting.m_Color[LID].y, gLighting.m_Color[LID].z);

					
					Vec3f P	= ToVec3f(gLighting.m_P[LID]);
					Vec3f N = ToVec3f(gLighting.m_N[LID]);
					Vec3f U = ToVec3f(gLighting.m_U[LID]);
					Vec3f V = ToVec3f(gLighting.m_V[LID]);
					Vec2f Luv = Vec2f(gLighting.m_Size[LID].x, gLighting.m_Size[LID].y);

					if (IntersectPlane(R, true, P, N, U, V, Luv, &T, pUV))
					{
						R.m_MaxT = T;
					}

 					if (DotN < 0.0f)
						L = m_Color / m_Area;
 					else
 						L = SPEC_BLACK;

					Le = ColorXYZf(gLighting.m_Color[LightID].x, gLighting.m_Color[LightID].y, gLighting.m_Color[LightID].z);

					if (pPdf)
						*pPdf = 1.0f;//DistanceSquared(R.m_O, Pl) / (DotN * m_Area);

					return true;
					*/

					break;
				}

				// Box
				case 1:
				{
					break;
				}

				// Sphere
				case 2:
				{
					Pl	= ToVec3f(gLighting.m_P[LID]) + UniformSampleSphere(RNG.Get2()) * ToVec3f(gLighting.m_Size[LID]);
					Pdf	= (Pe - Pl).Length();

					return ColorXYZf(gLighting.m_Color[LID].x, gLighting.m_Color[LID].y, gLighting.m_Color[LID].z);
				}
			}
		}
		
		// Sample background light
		case 1:
		{
			Pl	= BACKGROUND_LIGHT_RADIUS * UniformSampleSphere(RNG.Get2());
			Pdf	= (Pe - Pl).Length();

			return ColorXYZf(gLighting.m_Color[LID].x, gLighting.m_Color[LID].y, gLighting.m_Color[LID].z);
		}
	}
}

DEV bool HitTestLight(int LightID, CRay& R, float& T, ColorXYZf& Le, Vec2f* pUV = NULL, float* pPdf = NULL)
{
	switch (gLighting.m_Type[LightID])
	{
		// Intersect with area light
		case 0:
		{
			switch (gLighting.m_ShapeType[LightID])
			{
				// Plane
				case 0:
				{
					/*
					Vec3f P	= ToVec3f(gLighting.m_P[LightID]);
					Vec3f N = ToVec3f(gLighting.m_N[LightID]);
					Vec3f U = ToVec3f(gLighting.m_U[LightID]);
					Vec3f V = ToVec3f(gLighting.m_V[LightID]);
					Vec2f Luv = Vec2f(gLighting.m_Size[LightID].x, gLighting.m_Size[LightID].y);

					if (IntersectPlane(R, true, P, N, U, V, Luv, &T, pUV))
					{
						R.m_MaxT = T;
					}

 					if (DotN < 0.0f)
						L = m_Color / m_Area;
 					else
 						L = SPEC_BLACK;

					Le = ColorXYZf(gLighting.m_Color[LightID].x, gLighting.m_Color[LightID].y, gLighting.m_Color[LightID].z);

					if (pPdf)
						*pPdf = 1.0f;//DistanceSquared(R.m_O, Pl) / (DotN * m_Area);

					return true;
					*/

					break;
				}

				// Box
				case 1:
				{
					break;
				}

				// Sphere
				case 2:
				{
					if (IntersectSphere(R, ToVec3f(gLighting.m_Size[LightID]).x, &T))
						return true;
				}
			}

			break;
		}
		/**/

		// Intersect with background light
		case 1:
		{
			if (IntersectSphere(R, BACKGROUND_LIGHT_RADIUS, &T))
			{
				R.m_MaxT = T;
				Le = ColorXYZf(gLighting.m_Color[LightID].x, gLighting.m_Color[LightID].y, gLighting.m_Color[LightID].z);
				
				if (pPdf)
					*pPdf = 1.0f;//powf(BACKGROUND_LIGHT_RADIUS, 2.0f) / m_Area;

				return true;
			}
			else
			{
				return false;
			}
		}
	}

	return false;
}

DEV inline bool NearestLight(CRay R, ColorXYZf& LightColor, Vec3f& Pl, float* pPdf = NULL)
{
	bool Hit = false;
	
	float T = 0.0f;

	float Pdf = 0.0f;

	for (int LID = 0; LID < gLighting.m_NoLights; LID++)
	{
		if (HitTestLight(LID, R, T, LightColor, NULL, &Pdf))
		{
			Pl	= R(T);
			Hit	= true;
		}
	}
	
	if (pPdf)
		*pPdf = Pdf;

	return Hit;
}

DEV ColorXYZf EstimateDirectLight(CVolumeShader::EType Type, float Intensity, LightingSample& LS, Vec3f Wo, Vec3f Pe, Vec3f N, CRNG& RNG)
{
	ColorXYZf Ld = SPEC_BLACK, Li = SPEC_BLACK, F = SPEC_BLACK;
	
	CVolumeShader Shader(Type, N, Wo, GetDiffuse(Intensity), GetSpecular(Intensity), GetIOR(Intensity), GetGlossiness(Intensity));

	CRay Rl; 

	float LightPdf = 1.0f, ShaderPdf = 1.0f;

	Vec3f Wi, Pl;

 	Li = SampleLight(RNG, Pe, Pl, LightPdf);
	
	Rl.m_O		= Pe;
	Rl.m_D		= Normalize(Pe - Pl);
	Rl.m_MinT	= 0.0f;
	Rl.m_MaxT	= (Pe - Pl).Length();

	Wi = -Rl.m_D; 

	F = Shader.F(Wo, -Wi); 

	ShaderPdf = Shader.Pdf(Wo, -Wi);

	if (!Li.IsBlack() && ShaderPdf > 0.0f && LightPdf > 0.0f && !FreePathRM(Rl, RNG))
	{
		const float WeightMIS = PowerHeuristic(1.0f, LightPdf, 1.0f, ShaderPdf);
		
		if (Type == CVolumeShader::Brdf)
			Ld += F * Li * AbsDot(Wi, N) * WeightMIS / LightPdf;

		if (Type == CVolumeShader::Phase)
			Ld += F * Li * WeightMIS / LightPdf;
	}

	/*
	F = Shader.SampleF(Wo, Wi, ShaderPdf, LS.m_BsdfSample);

	if (!F.IsBlack() && ShaderPdf > 0.0f)
	{
		if (NearestLight(pScene, CRay(Pe, Wi, 0.0f), Li, Pl, pLight, &LightPdf))
		{
			LightPdf = pLight->Pdf(Pe, Wi);

			if (LightPdf > 0.0f && !Li.IsBlack() && !FreePathRM(CRay(Pl, Normalize(Pe - Pl), 0.0f, (Pe - Pl).Length()), RNG)) 
			{
				const float WeightMIS = PowerHeuristic(1.0f, ShaderPdf, 1.0f, LightPdf);

				if (Type == CVolumeShader::Brdf)
					Ld += F * Li * AbsDot(Wi, N) * WeightMIS / ShaderPdf;

				if (Type == CVolumeShader::Phase)
					Ld += F * Li * WeightMIS / ShaderPdf;
			}
		}
	}
	*/

	return Ld;
}

DEV ColorXYZf UniformSampleOneLight(CVolumeShader::EType Type, float Intensity, Vec3f Wo, Vec3f Pe, Vec3f N, CRNG& RNG)
{
	/*
	const int NumLights = pScene->m_Lighting.m_NoLights;

 	if (NumLights == 0)
 		return SPEC_BLACK;

	ColorXYZf Li;

	CLightingSample LS;

	LS.LargeStep(RNG);

	const int WhichLight = (int)floorf(LS.m_LightNum * (float)NumLights);

	CLight& Light = pScene->m_Lighting.m_Lights[WhichLight];
	
	return NumLights * EstimateDirectLight(pScene, Type, Density, Light, LS, Wo, Pe, N, RNG);
	*/

	LightingSample LS;

	LS.LargeStep(RNG);

	return EstimateDirectLight(Type, Intensity, LS, Wo, Pe, N, RNG);
}