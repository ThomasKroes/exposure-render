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

#include "MonteCarlo.cuh"
#include "Sample.cuh"

#define MAX_NO_LIGHTS 32

class EXPOSURE_RENDER_DLL CLight
{
public:
	float		m_Theta;
	float		m_Phi;
	float		m_Width;
	float		m_InvWidth;
	float		m_HalfWidth;
	float		m_InvHalfWidth;
	float		m_Height;
	float		m_InvHeight;
	float		m_HalfHeight;
	float		m_InvHalfHeight;
	float		m_Distance;
	float		m_SkyRadius;
	Vec3f		m_P;
	Vec3f		m_Target;	
	Vec3f		m_N;					
	Vec3f		m_U;					
	Vec3f		m_V;					
	float		m_Area;					
	float		m_AreaPdf;
	ColorXYZf	m_Color;
	ColorXYZf	m_ColorTop;
	ColorXYZf	m_ColorMiddle;
	ColorXYZf	m_ColorBottom;
	int			m_T;

	CLight(void) :
		m_Theta(0.0f),
		m_Phi(0.0f),
		m_Width(1.0f),
		m_InvWidth(1.0f / m_Width),
		m_HalfWidth(0.5f * m_Width),
		m_InvHalfWidth(1.0f / m_HalfWidth),
		m_Height(1.0f),
		m_InvHeight(1.0f / m_Height),
		m_HalfHeight(0.5f * m_Height),
		m_InvHalfHeight(1.0f / m_HalfHeight),
		m_Distance(1.0f),
		m_SkyRadius(100.0f),
		m_P(1.0f, 1.0f, 1.0f),
		m_Target(0.0f, 0.0f, 0.0f),
		m_N(1.0f, 0.0f, 0.0f),
		m_U(1.0f, 0.0f, 0.0f),
		m_V(1.0f, 0.0f, 0.0f),
		m_Area(m_Width * m_Height),
		m_AreaPdf(1.0f / m_Area),
		m_Color(10.0f),
		m_ColorTop(10.0f),
		m_ColorMiddle(10.0f),
		m_ColorBottom(10.0f),
		m_T(0)
	{
	}

	HOD CLight& operator=(const CLight& Other)
	{
		m_Theta				= Other.m_Theta;
		m_Phi				= Other.m_Phi;
		m_Width				= Other.m_Width;
		m_InvWidth			= Other.m_InvWidth;
		m_HalfWidth			= Other.m_HalfWidth;
		m_InvHalfWidth		= Other.m_InvHalfWidth;
		m_Height			= Other.m_Height;
		m_InvHeight			= Other.m_InvHeight;
		m_HalfHeight		= Other.m_HalfHeight;
		m_InvHalfHeight		= Other.m_InvHalfHeight;
		m_Distance			= Other.m_Distance;
		m_SkyRadius			= Other.m_SkyRadius;
		m_P					= Other.m_P;
		m_Target			= Other.m_Target;
		m_N					= Other.m_N;
		m_U					= Other.m_U;
		m_V					= Other.m_V;
		m_Area				= Other.m_Area;
		m_AreaPdf			= Other.m_AreaPdf;
		m_Color				= Other.m_Color;
		m_ColorTop			= Other.m_ColorTop;
		m_ColorMiddle		= Other.m_ColorMiddle;
		m_ColorBottom		= Other.m_ColorBottom;
		m_T					= Other.m_T;

		return *this;
	}

	HOD void Update(const CBoundingBox& BoundingBox)
	{
		m_InvWidth		= 1.0f / m_Width;
		m_HalfWidth		= 0.5f * m_Width;
		m_InvHalfWidth	= 1.0f / m_HalfWidth;
		m_InvHeight		= 1.0f / m_Height;
		m_HalfHeight	= 0.5f * m_Height;
		m_InvHalfHeight	= 1.0f / m_HalfHeight;
		m_Target		= BoundingBox.GetCenter();

		// Determine light position
		m_P.x = m_Distance * cosf(m_Phi) * sinf(m_Theta);
		m_P.z = m_Distance * cosf(m_Phi) * cosf(m_Theta);
		m_P.y = m_Distance * sinf(m_Phi);

		m_P += m_Target;

		// Determine area
		if (m_T == 0)
		{
			m_Area		= m_Width * m_Height;
			m_AreaPdf	= 1.0f / m_Area;
		}

		if (m_T == 1)
		{
			m_P				= BoundingBox.GetCenter();
			m_SkyRadius		= 1000.0f * (BoundingBox.GetMaxP() - BoundingBox.GetMinP()).Length();
			m_Area			= 4.0f * PI_F * powf(m_SkyRadius, 2.0f);
			m_AreaPdf		= 1.0f / m_Area;
		}

		// Compute orthogonal basis frame
		m_N = Normalize(m_Target - m_P);
		m_U	= Normalize(Cross(m_N, Vec3f(0.0f, 1.0f, 0.0f)));
		m_V	= Normalize(Cross(m_N, m_U));
	}

	// Samples the light
	HOD ColorXYZf SampleL(const Vec3f& P, CRay& Rl, float& Pdf, CLightingSample& LS)
	{
		ColorXYZf L = SPEC_BLACK;

		if (m_T == 0)
		{
			Rl.m_O	= m_P + ((-0.5f + LS.m_LightSample.m_Pos.x) * m_Width * m_U) + ((-0.5f + LS.m_LightSample.m_Pos.y) * m_Height * m_V);
			Rl.m_D	= Normalize(P - Rl.m_O);
			L		= Dot(Rl.m_D, m_N) > 0.0f ? Le(Vec2f(0.0f)) : SPEC_BLACK;
			Pdf		= AbsDot(Rl.m_D, m_N) > 0.0f ? DistanceSquared(P, Rl.m_O) / (AbsDot(Rl.m_D, m_N) * m_Area) : 0.0f;
		}

		if (m_T == 1)
		{
			Rl.m_O	= m_P + m_SkyRadius * UniformSampleSphere(LS.m_LightSample.m_Pos);
			Rl.m_D	= Normalize(P - Rl.m_O);
			L		= Le(Vec2f(1.0f) - 2.0f * LS.m_LightSample.m_Pos);
			Pdf		= powf(m_SkyRadius, 2.0f) / m_Area;
		}

		Rl.m_MinT	= 0.0f;
		Rl.m_MaxT	= (P - Rl.m_O).Length();

		return L;
	}

	// Intersect ray with light
	HOD bool Intersect(CRay& R, float& T, ColorXYZf& L, Vec2f* pUV = NULL, float* pPdf = NULL)
	{
		if (m_T == 0)
		{
			// Compute projection
			const float DotN = Dot(R.m_D, m_N);

			// Rays is co-planar with light surface
			if (DotN >= 0.0f)
				return false;

			// Compute hit distance
			T = (-m_Distance - Dot(R.m_O, m_N)) / DotN;

			// Intersection is in ray's negative direction
			if (T < R.m_MinT || T > R.m_MaxT)
				return false;

			// Determine position on light
			const Vec3f Pl = R(T);

			// Vector from point on area light to center of area light
			const Vec3f Wl = Pl - m_P;

			// Compute texture coordinates
			const Vec2f UV = Vec2f(Dot(Wl, m_U), Dot(Wl, m_V));

			// Check if within bounds of light surface
			if (UV.x > m_HalfWidth || UV.x < -m_HalfWidth || UV.y > m_HalfHeight || UV.y < -m_HalfHeight)
				return false;

			R.m_MaxT = T;

			if (pUV)
				*pUV = UV;

 			if (DotN < 0.0f)
				L = m_Color / m_Area;
 			else
 				L = SPEC_BLACK;

			if (pPdf)
				*pPdf = DistanceSquared(R.m_O, Pl) / (DotN * m_Area);

			return true;
		}

		if (m_T == 1)
		{
			T = m_SkyRadius;
			
			// Intersection is in ray's negative direction
			if (T < R.m_MinT || T > R.m_MaxT)
				return false;
			
			R.m_MaxT = T;

			Vec2f UV = Vec2f(SphericalPhi(R.m_D) * INV_TWO_PI_F, SphericalTheta(R.m_D) * INV_PI_F);

			L	= Le(Vec2f(1.0f) - 2.0f * UV);

			if (pPdf)
				*pPdf = powf(m_SkyRadius, 2.0f) / m_Area;

			return true;
		}

		return false;
	}

	HOD float Pdf(const Vec3f& P, const Vec3f& Wi)
	{
		ColorXYZf L;
		Vec2f UV;
		float Pdf = 1.0f;

		CRay Rl = CRay(P, Wi, 0.0f, INF_MAX);

		if (m_T == 0)
		{
			float T = 0.0f;
			
			if (!Intersect(Rl, T, L, NULL, &Pdf))
				return 0.0f;

			return powf(T, 2.0f) / (AbsDot(m_N, -Wi) * m_Area);
		}

		if (m_T == 1)
		{
			return powf(m_SkyRadius, 2.0f) / m_Area;
		}

		return 0.0f;
	}

	HOD ColorXYZf Le(const Vec2f& UV)
	{
		if (m_T == 0)
			return m_Color / m_Area;

		if (m_T == 1)
		{
			if (UV.y > 0.0f)
				return Lerp(fabs(UV.y), m_ColorMiddle, m_ColorTop);
			else
				return Lerp(fabs(UV.y), m_ColorMiddle, m_ColorBottom);
		}

		return SPEC_BLACK;
	}
};

class EXPOSURE_RENDER_DLL CLighting
{
public:
	CLighting(void) :
		m_NoLights(0)
	{
	}

	HOD CLighting& operator=(const CLighting& Other)
	{
		for (int i = 0; i < MAX_NO_LIGHTS; i++)
		{
			m_Lights[i] = Other.m_Lights[i];
		}

		m_NoLights = Other.m_NoLights;

		return *this;
	}

	void AddLight(const CLight& Light)
	{
// 		if (m_NoLights >= MAX_NO_LIGHTS)
// 			return;

		m_Lights[m_NoLights] = Light;

		m_NoLights = m_NoLights + 1;
	}

	void Reset(void)
	{
		m_NoLights = 0;
		//memset(m_Lights, 0 , MAX_NO_LIGHTS * sizeof(CLight));
	}

	CLight		m_Lights[MAX_NO_LIGHTS];
	int			m_NoLights;
};

DEV bool NearestLight(CRay R, ColorXYZf& LightColor, Vec3f& Pl, CLight*& pLight, float* pPdf = NULL)
{
	bool Hit = false;
	
	/*
	float T = 0.0f;

	CRay RayCopy = R;

	float Pdf = 0.0f;

	for (int i = 0; i < pScene->m_Lighting.m_NoLights; i++)
	{
		if (pScene->m_Lighting.m_Lights[i].Intersect(RayCopy, T, LightColor, NULL, &Pdf))
		{
			Pl		= R(T);
			pLight	= &pScene->m_Lighting.m_Lights[i];
			Hit		= true;
		}
	}
	
	if (pPdf)
		*pPdf = Pdf;
	*/

	return Hit;
}