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

DEV Vec3f TransformVector(TransformMatrix TM, Vec3f v)
{
  Vec3f r;
//  r.x = Dot(v, Vec3f(TM.NN[0][0], TM.NN[1][0], TM.NN[2][0]));
//  r.y = Dot(v, Vec3f(TM.NN[0][1], TM.NN[1][1], TM.NN[2][1]));
//  r.z = Dot(v, Vec3f(TM.NN[0][2], TM.NN[1][2], TM.NN[2][2]));

  float x = v.x, y = v.y, z = v.z;

  r.x = TM.NN[0][0] * x + TM.NN[0][1] * y + TM.NN[0][2] * z;
  r.y = TM.NN[1][0] * x + TM.NN[1][1] * y + TM.NN[1][2] * z;
  r.z = TM.NN[2][0] * x + TM.NN[2][1] * y + TM.NN[2][2] * z;

  return r;
}

DEV Vec3f TransformPoint(TransformMatrix TM, Vec3f pt)
{
	/*
	float x = pt.x, y = pt.y, z = pt.z;
    ptrans->x = m.m[0][0]*x + m.m[0][1]*y + m.m[0][2]*z + m.m[0][3];
    ptrans->y = m.m[1][0]*x + m.m[1][1]*y + m.m[1][2]*z + m.m[1][3];
    ptrans->z = m.m[2][0]*x + m.m[2][1]*y + m.m[2][2]*z + m.m[2][3];
    float w   = m.m[3][0]*x + m.m[3][1]*y + m.m[3][2]*z + m.m[3][3];
    if (w != 1.) *ptrans /= w;
	*/
    float x = pt.x, y = pt.y, z = pt.z;
    float xp = TM.NN[0][0]*x + TM.NN[0][1]*y + TM.NN[0][2]*z + TM.NN[0][3];
    float yp = TM.NN[1][0]*x + TM.NN[1][1]*y + TM.NN[1][2]*z + TM.NN[1][3];
    float zp = TM.NN[2][0]*x + TM.NN[2][1]*y + TM.NN[2][2]*z + TM.NN[2][3];
//    float wp = TM.NN[3][0]*x + TM.NN[3][1]*y + TM.NN[3][2]*z + TM.NN[3][3];
    
//	Assert(wp != 0);
    
//	if (wp == 1.)
		return Vec3f(xp, yp, zp);
 //   else
//		return Vec3f(xp, yp, zp) * (1.0f / wp);
}

DEV CRay TransformRay(CRay R, TransformMatrix TM)
{
	CRay TR;

	Vec3f O(TM.NN[0][3], TM.NN[1][3], TM.NN[2][3]);

	TR.m_O.x	= Dot(TR.m_O - O, Vec3f(TM.NN[0][0], TM.NN[1][0], TM.NN[2][0]));
	TR.m_O.y	= Dot(TR.m_O - O, Vec3f(TM.NN[0][1], TM.NN[1][1], TM.NN[2][1]));
	TR.m_O.z	= Dot(TR.m_O - O, Vec3f(TM.NN[0][2], TM.NN[1][2], TM.NN[2][2]));

	TR.m_O = TransformPoint(TM, R.m_O);

	TR.m_D.x	= Dot(TR.m_D, Vec3f(TM.NN[0][0], TM.NN[0][1], TM.NN[0][2]));
	TR.m_D.y	= Dot(TR.m_D, Vec3f(TM.NN[1][0], TM.NN[1][1], TM.NN[1][2]));
	TR.m_D.z	= Dot(TR.m_D, Vec3f(TM.NN[2][0], TM.NN[2][1], TM.NN[2][2]));

	TR.m_D = TransformVector(TM, R.m_D);

	TR.m_MinT	= R.m_MinT;
	TR.m_MaxT	= R.m_MaxT;

	return TR;
}

DEV ColorXYZf SampleLight(CRNG& RNG, const Vec3f& Pe, Vec3f& Pl, float& Pdf)
{
	const int LID = floorf(RNG.Get1() * gLighting.m_NoLights);

	Light& L = gLighting.m_Lights[LID];

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
			LocalP = BACKGROUND_LIGHT_RADIUS * SampleUnitSphere(RNG.Get2());
			break;
		}
	}

	Pl	= TransformPoint(L.m_TM, LocalP);
	Pdf	= DistanceSquared(Pe, Pl) / L.m_Area;

	return ColorXYZf(L.m_Color.x, L.m_Color.y, L.m_Color.z) / L.m_Area;
}

DEV bool HitTestLight(int LightID, CRay& R, float& T, ColorXYZf& Le, Vec2f* pUV = NULL, float* pPdf = NULL)
{
	Light& L = gLighting.m_Lights[LightID];

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
					Res = IntersectPlane(TR, L.m_OneSided, ToVec3f(L.m_Size), &Tl, NULL);
					break;
				}

				// Disk
				case 1:
				{
					Res = IntersectDisk(TR, L.m_OneSided, L.m_OuterRadius, &Tl, NULL);
					break;
				}

				// Ring
				case 2:
				{
					Res = IntersectRing(TR, L.m_OneSided, L.m_InnerRadius, L.m_OuterRadius, &Tl, NULL);
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
			Res = IntersectSphere(TR, BACKGROUND_LIGHT_RADIUS, &Tl);
			break;
		}
	}

	if (Res > 0)
	{
		Le = Res == 1 ? ColorXYZf(L.m_Color.x, L.m_Color.y, L.m_Color.z) : ColorXYZf(0.0f);
		
		// Compute intersection point in world coordinates
		const Vec3f P = TransformPoint(L.m_TM, TR(Tl));

		// Compute worl distance to intersection
		T = Length(P - R.m_D);

		*pPdf = 1.0f;

		return true;
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