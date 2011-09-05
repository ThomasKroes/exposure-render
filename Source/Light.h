#pragma once

#include "Geometry.h"
#include "Random.h"
#include "MonteCarlo.h"



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

	/*
	// ToDo: Add description
	DEV void RegisterHit(const float& RayEpsilon, const CHitRecord& HitRecord, Vec3f* pV, Vec4i* pVi, Vec3f* pVN, Vec4i* pVNi, Vec2f* pUV, Vec4i* pUVi)
	{
		// Set primitive and object ID
		m_PrimitiveID	= HitRecord.m_FaceID;
		m_ObjectID		= pVi[m_PrimitiveID].w;

		// Vertex, vertex normal and texture coordinate indices
		const Vec4i Vi	= pVi[m_PrimitiveID];
		const Vec4i Ni	= pVNi[m_PrimitiveID];
		const Vec4i UVi	= pUVi[m_PrimitiveID];

		// Vertex positions
		const Vec3f P0	= pV[Vi.x];
		const Vec3f P1	= pV[Vi.y];
		const Vec3f P2	= pV[Vi.z];

		// Texture coordinates
		const Vec2f UV0	= pUV[UVi.x];
		const Vec2f UV1	= pUV[UVi.y];
		const Vec2f UV2	= pUV[UVi.z];

		// Normals
		const Vec3f N0	= pVN[Ni.x];
		const Vec3f N1	= pVN[Ni.y];
		const Vec3f N2	= pVN[Ni.z];

		// Compute texture coordinates, normal and hit position
 		m_UV	= Vec2f(HitRecord.m_B.x * UV0.x + HitRecord.m_B.y * UV1.x + HitRecord.m_B.z * UV2.x, HitRecord.m_B.x * UV0.y + HitRecord.m_B.y * UV1.y + HitRecord.m_B.z * UV2.y);
		m_Ng	= Normalize(HitRecord.m_B.x * N0 + HitRecord.m_B.y * N1 + HitRecord.m_B.z * N2);
		m_P		= (HitRecord.m_B.x * P0 + HitRecord.m_B.y * P1 + HitRecord.m_B.z * P2) + RayEpsilon * m_Ng;

		CreateCS(m_Ng, m_NU, m_NV);

		return;

		/*
		// calculate dPdU and dPdV
		float du1 = UV0.x - UV2.x;
		float du2 = UV1.x - UV2.x;
		float dv1 = UV0.y - UV2.y;
		float dv2 = UV1.y - UV2.y;
		float invdet, det = du1 * dv2 - dv1 * du2;

		if(fabs(det) > 1e-30f)
		{
			invdet = 1.0f / det;
			
			Vec3f dp1 = P0 - P2;
			Vec3f dp2 = P1 - P2;

			m_dPdU = (dv2 * invdet) * dp1 - (dv1 * invdet) * dp2;
			m_dPdV = (du1 * invdet) * dp2 - (du2 * invdet) * dp1;
		}
		else
		{
			m_dPdU = Vec3f(0.f);
			m_dPdV = Vec3f(0.f);
		}

		CreateCS(m_Ng, m_NU, m_NV);

		// Transform dPdU and dPdV in shading space
		m_dSdU.x = Dot(m_NU, m_dPdU);
		m_dSdU.y = Dot(m_NV, m_dPdU);
		m_dSdU.z = Dot(m_Ng, m_dPdU);
		m_dSdV.x = Dot(m_NU, m_dPdV);
		m_dSdV.y = Dot(m_NV, m_dPdV);
		m_dSdV.z = Dot(m_Ng, m_dPdV);
		
	}
*/
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
	enum EType
	{
		Area,
		Background
	};

	EType			m_Type;
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
	Vec3f			m_P;
	Vec3f			m_Target;	
	Vec3f			m_N;					
	Vec3f			m_U;					
	Vec3f			m_V;					
	float			m_Area;					
	float			m_Pdf;
	CColorRgbHdr	m_Color;

	CLight(void) :
		m_Type(Area),
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
		m_P(1.0f, 1.0f, 1.0f),
		m_Target(0.0f, 0.0f, 0.0f),
		m_N(1.0f, 0.0f, 0.0f),
		m_U(1.0f, 0.0f, 0.0f),
		m_V(1.0f, 0.0f, 0.0f),
		m_Area(m_Width * m_Height),
		m_Pdf(1.0f / m_Area),
		m_Color(10.0f)
	{
	}

	HOD CLight& operator=(const CLight& Other)
	{
		m_Type				= Other.m_Type;
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
		m_P					= Other.m_P;
		m_Target			= Other.m_Target;
		m_N					= Other.m_N;
		m_U					= Other.m_U;
		m_V					= Other.m_V;
		m_Area				= Other.m_Area;
		m_Pdf				= Other.m_Pdf;
		m_Color				= Other.m_Color;

		return *this;
	}

	HOD void Update(void)
	{
		m_InvWidth		= 1.0f / m_Width;
		m_HalfWidth		= 0.5f * m_Width;
		m_InvHalfWidth	= 1.0f / m_HalfWidth;
		m_InvHeight		= 1.0f / m_Height;
		m_HalfHeight	= 0.5f * m_Height;
		m_InvHalfHeight	= 1.0f / m_HalfHeight;

		// Determine light position
		m_P.x = m_Distance * cosf(m_Phi / RAD_F) * sinf(m_Theta / RAD_F);
		m_P.z = m_Distance * cosf(m_Phi / RAD_F) * cosf(m_Theta / RAD_F);
		m_P.y = m_Distance * sinf(m_Phi / RAD_F);

		m_P += m_Target;

		// Determine area
		m_Area = m_Width * m_Height;

		// Determine pdf
		m_Pdf = 1.0f;

		// Compute orthogonal basis frame
		m_N = Normalize(m_Target - m_P);
		m_U	= Normalize(Cross(m_N, Vec3f(0.0f, 1.0f, 0.0f)));
		m_V	= Normalize(Cross(m_N, m_U));
	}

	// Samples the light
	HOD CColorXyz SampleL(CSurfacePoint& SP, CSurfacePoint& SPl, CLightingSample& LS, float& Pdf, const float& RayEpsilon)
	{
		switch (m_Type)
		{
			case Area:
			{
				// Compute light ray position and direction
				SPl.m_P	= m_P + ((-0.5f + LS.m_LightSample.m_Pos.x) * m_Width * m_U) + ((-0.5f + LS.m_LightSample.m_Pos.y) * m_Height * m_V);

				const Vec3f Wi = SP.m_P - SPl.m_P;

				// Compute probability
				Pdf = Dot(Wi, m_N) > 0.0f ? DistanceSquared(SP.m_P, SPl.m_P) / (AbsDot(Wi, m_N) * m_Area) : 0.0f;

				// Set the light color
				return Dot(Wi, m_N) > 0.0f ? Le(Vec2f(0.0f)) : SPEC_BLACK;
			}

			case Background:
			{
				UniformSampleSphere(LS.m_LightSample.m_Pos);
			}
		}
	}

	// Intersect ray with light
	HOD bool Intersect(const Vec3f& P, const Vec3f& W, const float& MinT, const float& MaxT, float& T, bool* pFront = NULL, Vec2f* pUV = NULL, float* pPdf = NULL)
	{
		switch (m_Type)
		{
			case Area:
			{
				// Intersection ray
				CRay R(P, W, MinT, MaxT);

				// Compute projection
				const float DotN = Dot(R.m_D, m_N);

				// Rays is co-planar with light surface
				if (DotN == 0.0f)
					return false;

				// Compute hit distance
				T = (-m_Distance - Dot(R.m_O, m_N)) / DotN;

				// Intersection is in ray's negative direction
				if (T < MinT || T > MaxT)
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

				if (pUV)
					*pUV = UV;

				if (pFront)
					*pFront = DotN < 0.0f;

				if (pPdf)
					*pPdf = DistanceSquared(P, Pl) / m_Area;

				return true;
			}

			case Background:
			{
				return true;
			}
		}
	}

	HOD float Pdf(const Vec3f& P, const Vec3f& Wi)
	{
		switch (m_Type)
		{
			case Area:
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

			case Background:
			{
				return 1.0f;
			}
		}
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
		for (int i = 0; i < m_NoLights; i++)
		{
			m_Lights[i] = Other.m_Lights[i];
		}

		m_NoLights = Other.m_NoLights;

		return *this;
	}

	CLight			m_Lights[MAX_NO_LIGHTS];
	int				m_NoLights;
};