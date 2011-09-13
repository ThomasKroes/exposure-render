#pragma once

#include "MonteCarlo.h"
#include "Sample.h"

#define MAX_NO_LIGHTS 32

class CSurfacePoint
{
public:
	Vec3f			m_Ns;				/*!< Shading normal at surface point */
	Vec3f			m_Ng;				/*!< Geometric normal at surface point */
	Vec3f			m_P;				/*!< World position of the surface point */
	bool			m_HasUV;			/*!< Determines whether UV coordinates are known at the surface point */
	int				m_PrimitiveID;		/*!< ID of the primitive at the surface point */
	int				m_ObjectID;			/*!< ID of the object at the surface point */
	Vec2f			m_UV;				/*!< UV texture coordinate at the surface point */
	Vec3f  			m_NU; 				/*!< Second vector building orthogonal shading space with N */
	Vec3f  			m_NV; 				/*!< Third vector building orthogonal shading space with N */
	Vec3f 			m_dPdU; 			/*!< u-axis in world space */
	Vec3f 			m_dPdV; 			/*!< v-axis in world space */
	Vec3f 			m_dSdU; 			/*!< u-axis in shading space (NU, NV, N) */
	Vec3f 			m_dSdV; 			/*!< v-axis in shading space (NU, NV, N) */
	float			m_sU; 				/*!< Raw surface parametric coordinate; required to evaluate vmaps */
	float			m_sV; 				/*!< Raw surface parametric coordinate; required to evaluate vmaps */
	Vec2f			m_DuDv;				/*!< Raw surface parametric coordinate; required to evaluate vmaps */

	// ToDo: Add description
	HOD CSurfacePoint(void)
	{
		m_Ns			= Vec3f(0.0f);
		m_Ng			= Vec3f(0.0f);
		m_P				= Vec3f(0.0f);
		m_HasUV			= false;
		m_PrimitiveID	= -1;
		m_ObjectID		= -1;
		m_UV			= Vec2f(0.0f);			
		m_NU			= Vec3f(0.0f);			
		m_NV			= Vec3f(0.0f);				
		m_dPdU			= Vec3f(0.0f);
		m_dPdV			= Vec3f(0.0f);
		m_dSdU			= Vec3f(0.0f);
		m_dSdV			= Vec3f(0.0f);
		m_sU			= 0.0f;
		m_sV			= 0.0f;			
		m_DuDv			= Vec2f(0.0f);	
	}

	// ToDo: Add description
	HOD ~CSurfacePoint(void)
	{
	}

	// ToDo: Add description
	DEV CSurfacePoint& CSurfacePoint::operator=(const CSurfacePoint& Other)
	{
		m_Ns			= Other.m_Ns;			
		m_Ng			= Other.m_Ng;	
		m_P				= Other.m_P;			
		m_HasUV			= Other.m_HasUV;		
		m_PrimitiveID	= Other.m_PrimitiveID;		
		m_ObjectID		= Other.m_ObjectID;		
		m_UV			= Other.m_UV;			
		m_NU			= Other.m_NU; 			
		m_NV			= Other.m_NV; 			
		m_dPdU			= Other.m_dPdU; 		
		m_dPdV			= Other.m_dPdV; 		
		m_dSdU			= Other.m_dSdU; 		
		m_dSdV			= Other.m_dSdV; 		
		m_sU			= Other.m_sU; 			
		m_sV			= Other.m_sV; 

		return *this;
	}

	
	// ToDo: Add description
	DEV void ComputeBump(void)
	{
		m_NU += m_DuDv.x * m_Ng;
		m_NV += m_DuDv.y * m_Ng;
		m_Ng = Normalize(Cross(m_NU, m_NV));
		m_NU.Normalize();
		m_NV = Cross(m_Ng, m_NU);
	}
};

class CLight
{
public:
	float			m_Theta;
	float			m_Phi;
	float			m_Width;
	float			m_InvWidth;
	float			m_HalfWidth;
	float			m_InvHalfWidth;
	float			m_Height;
	float			m_InvHeight;
	float			m_HalfHeight;
	float			m_InvHalfHeight;
	float			m_Distance;
	float			m_SkyRadius;
	Vec3f			m_P;
	Vec3f			m_Target;	
	Vec3f			m_N;					
	Vec3f			m_U;					
	Vec3f			m_V;					
	float			m_Area;					
	float			m_AreaPdf;
	CColorRgbHdr	m_Color;
	CColorRgbHdr	m_ColorTop;
	CColorRgbHdr	m_ColorMiddle;
	CColorRgbHdr	m_ColorBottom;
	int				m_T;

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
			m_Area		= 4.0f * PI_F * powf(m_SkyRadius, 2.0f);
			m_AreaPdf	= 1.0f / m_Area;
		}

		// Compute orthogonal basis frame
		m_N = Normalize(m_Target - m_P);
		m_U	= Normalize(Cross(m_N, Vec3f(0.0f, 1.0f, 0.0f)));
		m_V	= Normalize(Cross(m_N, m_U));
	}

	// Samples the light
	HOD CColorXyz SampleL(const Vec3f& P, CRay& Rl, float& Pdf, CLightingSample& LS)
	{
		// Exitant radiance
		CColorXyz L = SPEC_BLACK;

		// Determine position on light
		if (m_T == 0)
		{
			Rl.m_O	= m_P + ((-0.5f + LS.m_LightSample.m_Pos.x) * m_Width * m_U) + ((-0.5f + LS.m_LightSample.m_Pos.y) * m_Height * m_V);
			Rl.m_D	= Normalize(P - Rl.m_O);
			L		= Le(Vec2f(0.0f));
			Pdf		= DistanceSquared(P, Rl.m_O) / AbsDot(Rl.m_D, m_N) * m_Area;
		}

		if (m_T == 1)
		{
			Rl.m_O	= m_Target + m_SkyRadius * UniformSampleSphere(LS.m_LightSample.m_Pos);
			Rl.m_D	= Normalize(P - Rl.m_O);
			L		= Le(Vec2f(0.0f));
			Pdf		= DistanceSquared(P, Rl.m_O) / m_Area;
		}

		Rl.m_MinT	= 0.0f;
		Rl.m_MaxT	= (P - Rl.m_O).Length();

		return L;
	}

	// Intersect ray with light
	HOD bool Intersect(CRay& R, float& T, CColorXyz& Le, Vec2f* pUV = NULL, float* pPdf = NULL)
	{
		if (m_T == 0)
		{
			// Compute projection
			const float DotN = Dot(R.m_D, m_N);

			// Rays is co-planar with light surface
			if (DotN == 0.0f)
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
				Le = m_Color.ToXYZ() / m_Area;
 			else
 				Le = SPEC_BLACK;

			if (pPdf)
				*pPdf = DistanceSquared(R.m_O, Pl) / m_Area;

			return true;
		}

		if (m_T == 1)
		{
			T = m_SkyRadius;
			
			// Intersection is in ray's negative direction
			if (T < R.m_MinT || T > R.m_MaxT)
				return false;
			
			R.m_MaxT	= T;

			Vec2f pUV = Vec2f(SphericalPhi(R.m_D) * INV_TWO_PI_F, SphericalTheta(R.m_D) * INV_PI_F);

// 			if (pUV.y > 0.0f)
// 				Le = Lerp(fabs(pUV.y), m_ColorTop.ToXYZ(), m_ColorMiddle.ToXYZ());
// 			else
// 				Le = Lerp(fabs(pUV.y), m_ColorTop.ToXYZ(), m_ColorMiddle.ToXYZ());

			Le = m_Color.ToXYZ();			

			return true;
		}

		return false;
	}

	HOD float Pdf(const Vec3f& P, const Vec3f& Wi)
	{
		/*
		switch (m_Type)
		{
			case 0:
			{
				// Hit distance
				float T = 0.0f;

				// Intersect ray with light surface
				if (!Intersect(P, Wi, 0.0f, INF_MAX, T))
					return INF_MAX;

				// Ray is exactly co-planar with area light surface
				if (AbsDot(m_N, -Wi) == 0.0f) 
					return INF_MAX;

				// Convert light sample weight to solid angle measure
				return T / (AbsDot(m_N, -Wi) * m_Area);
			}

			case 1:
			{
				return 1.0f;
			}

			default:
				return 1.0f;
		}
		*/

		return 1.0f;
	}

	HOD CColorXyz Le(const Vec2f& UV)
	{
		return CColorXyz::FromRGB(m_Color.r, m_Color.g, m_Color.b);
	}
};

class CLighting
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