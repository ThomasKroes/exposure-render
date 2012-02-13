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

#include <cuda_runtime.h>

#include "General.cuh"

DEV Vec3f TransformVector(_TransformMatrix TM, Vec3f v)
{
  Vec3f r;

  float x = v.x, y = v.y, z = v.z;

  r.x = TM.NN[0][0] * x + TM.NN[0][1] * y + TM.NN[0][2] * z;
  r.y = TM.NN[1][0] * x + TM.NN[1][1] * y + TM.NN[1][2] * z;
  r.z = TM.NN[2][0] * x + TM.NN[2][1] * y + TM.NN[2][2] * z;

  return r;
}

DEV Vec3f TransformPoint(_TransformMatrix TM, Vec3f pt)
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

DEV CRay TransformRay(CRay R, _TransformMatrix TM)
{
	CRay TR;

	TR.m_O 		= TransformPoint(TM, R.m_O);
	TR.m_D 		= TransformVector(TM, R.m_D);
	TR.m_MinT	= R.m_MinT;
	TR.m_MaxT	= R.m_MaxT;

	return TR;
}

HOD inline Vec3f ToVec3f(const float3& V)
{
	return Vec3f(V.x, V.y, V.z);
}

HOD inline Vec3f ToVec3f(float V[3])
{
	return Vec3f(V[0], V[1], V[2]);
}

HOD inline float3 FromVec3f(const Vec3f& V)
{
	return make_float3(V.x, V.y, V.z);
}

DEV float GetIntensity(Vec3f P)
{
	return (float)(USHRT_MAX * tex3D(gTexIntensity, (P.x - gVolume.m_MinAABB[0]) * gVolume.m_InvSize[0], (P.y - gVolume.m_MinAABB[1]) * gVolume.m_InvSize[1], (P.z - gVolume.m_MinAABB[2]) * gVolume.m_InvSize[2]));
}

DEV float GetNormalizedIntensity(Vec3f P)
{
	return (GetIntensity(P) - gVolume.m_IntensityMin) * gVolume.m_IntensityInvRange;
}

DEV float GetOpacity(float NormalizedIntensity)
{
	return tex1D(gTexOpacity, NormalizedIntensity);
}

DEV bool Inside(_ClippingObject& ClippingObject, Vec3f P)
{
	bool Inside = false;

	switch (ClippingObject.m_ShapeType)
	{
		// Plane
		case 0:
		{
			Inside = P.z > 0.0f;
			break;
		}

		// Box
		case 1:
		{
			const float HalfSize[3] = { 0.5f * ClippingObject.m_Size[0], 0.5f * ClippingObject.m_Size[1], 0.5f * ClippingObject.m_Size[2] };

			Inside = P.x > -HalfSize[0] && P.x < HalfSize[0] && P.y > -HalfSize[1] && P.y < HalfSize[1] && P.z > -HalfSize[2] && P.z < HalfSize[2];
			break;
		}

		// Sphere
		case 2:
		{
			Inside = Length(P) < ClippingObject.m_Radius;
			break;
		}

		// Cylinder
		case 3:
		{
			Inside = sqrtf((P.x * P.x) + (P.z * P.z)) < ClippingObject.m_Radius && fabs(P.y) < (0.5f * ClippingObject.m_Size[1]);
			break;
		}
	}

	return ClippingObject.m_Invert ? !Inside : Inside;
}

DEV bool Inside(Vec3f P)
{
	for (int i = 0; i < gClipping.m_NoClippingObjects; i++)
	{
		_ClippingObject& ClippingObject = gClipping.m_ClippingObjects[i];

		const Vec3f P2 = TransformPoint(ClippingObject.m_InvTM, P);

		if (Inside(ClippingObject, P2))
			return true;
	}

	return false;
}

DEV float GetOpacity(Vec3f P)
{
	const float Intensity = GetIntensity(P);
	
	for (int i = 0; i < gClipping.m_NoClippingObjects; i++)
	{
		_ClippingObject& ClippingObject = gClipping.m_ClippingObjects[i];

		const Vec3f P2 = TransformPoint(ClippingObject.m_TM, P);

//		const bool InRange = Intensity > ClippingObject.m_MinIntensity && Intensity < ClippingObject.m_MaxIntensity;

		if (Inside(ClippingObject, P2))
			return 0.0f;
	}

	const float NormalizedIntensity = (Intensity - gOpacityRange.m_Min) * gOpacityRange.m_InvRange;

	return GetOpacity(NormalizedIntensity);
}

DEV ColorXYZf GetDiffuse(float Intensity)
{
	const float NormalizedIntensity = (Intensity - gDiffuseRange.m_Min) * gDiffuseRange.m_InvRange;

	float4 Diffuse = tex1D(gTexDiffuse, NormalizedIntensity);
	return ColorXYZf(Diffuse.x, Diffuse.y, Diffuse.z);
}

DEV ColorXYZf GetSpecular(float Intensity)
{
	const float NormalizedIntensity = (Intensity - gSpecularRange.m_Min) * gSpecularRange.m_InvRange;

	float4 Specular = tex1D(gTexSpecular, NormalizedIntensity);
	return ColorXYZf(Specular.x, Specular.y, Specular.z);
}

DEV float GetGlossiness(float Intensity)
{
	const float NormalizedIntensity = (Intensity - gGlossinessRange.m_Min) * gGlossinessRange.m_InvRange;

	return tex1D(gTexGlossiness, NormalizedIntensity);
}

DEV float GetIor(float Intensity)
{
	const float NormalizedIntensity = (Intensity - gIorRange.m_Min) * gIorRange.m_InvRange;

	return tex1D(gTexIor, NormalizedIntensity);
}

DEV ColorXYZf GetEmission(float Intensity)
{
	const float NormalizedIntensity = (Intensity - gEmissionRange.m_Min) * gEmissionRange.m_InvRange;

	float4 Emission = tex1D(gTexEmission, NormalizedIntensity);
	return ColorXYZf(Emission.x, Emission.y, Emission.z);
}

DEV inline Vec3f NormalizedGradient(Vec3f P)
{
	Vec3f Gradient;

	Vec3f Pts[3][2];

	Pts[0][0] = P + ToVec3f(gVolume.m_GradientDeltaX);
	Pts[0][1] = P - ToVec3f(gVolume.m_GradientDeltaX);
	Pts[1][0] = P + ToVec3f(gVolume.m_GradientDeltaY);
	Pts[1][1] = P - ToVec3f(gVolume.m_GradientDeltaY);
	Pts[2][0] = P + ToVec3f(gVolume.m_GradientDeltaZ);
	Pts[2][1] = P - ToVec3f(gVolume.m_GradientDeltaZ);

	float Ints[3][2];

	//Ints[0][0] = Inside(Pts[0][0]) ? 0.0f : GetIntensity(Pts[0][0]);
	//Ints[0][1] = Inside(Pts[0][1]) ? 0.0f : GetIntensity(Pts[0][1]);
	//Ints[1][0] = Inside(Pts[1][0]) ? 0.0f : GetIntensity(Pts[1][0]);
	//Ints[1][1] = Inside(Pts[1][1]) ? 0.0f : GetIntensity(Pts[1][1]);
	//Ints[2][0] = Inside(Pts[2][0]) ? 0.0f : GetIntensity(Pts[2][0]);
	//Ints[2][1] = Inside(Pts[2][1]) ? 0.0f : GetIntensity(Pts[2][1]);

	Ints[0][0] = GetIntensity(Pts[0][0]);
	Ints[0][1] = GetIntensity(Pts[0][1]);
	Ints[1][0] = GetIntensity(Pts[1][0]);
	Ints[1][1] = GetIntensity(Pts[1][1]);
	Ints[2][0] = GetIntensity(Pts[2][0]);
	Ints[2][1] = GetIntensity(Pts[2][1]);

	Gradient.x = (Ints[0][1] - Ints[0][0]) * gVolume.m_InvGradientDelta;
	Gradient.y = (Ints[1][1] - Ints[1][0]) * gVolume.m_InvGradientDelta;
	Gradient.z = (Ints[2][1] - Ints[2][0]) * gVolume.m_InvGradientDelta;

	return Normalize(Gradient);
}

#define INTERSECTION_EPSILON 0.001f

DEV int IntersectPlane(CRay R, bool OneSided, float* pT = NULL, Vec2f* pUV = NULL)
{
	// Avoid floating point precision issues near parallel intersections
	if (fabs(R.m_O.z - R.m_D.z) < INTERSECTION_EPSILON)
		return 0;

	// Compute intersection distance
	const float T = (0.0f - R.m_O.z) / R.m_D.z;

	// Satisfy the ray's parametric range
	if (T < R.m_MinT || T > R.m_MaxT)
		return 0;

	Vec3f Pl;

	// Compute intersection point
	Pl.x = R.m_O.x + T * (R.m_D.x);
	Pl.y = R.m_O.y + T * (R.m_D.y);
	Pl.z = 0.0f;

	if (pT)
		*pT = T;

	if (pUV)
		*pUV = Vec2f(Pl.x, Pl.y);

	if (OneSided && R.m_D.z > 0.0f)
		return 2;
	else
		return 1;
}

DEV int IntersectUnitPlane(CRay R, bool OneSided, float* pT = NULL, Vec2f* pUV = NULL)
{
	Vec2f UV;

	int Res = IntersectPlane(R, OneSided, pT, &UV);

	if (Res <= 0)
		return 0;
	else
	{
		if (UV.x < -0.5f || UV.x > 0.5f || UV.y < -0.5f || UV.y > 0.5f)
			return 0;
	}

	if (pUV)
		*pUV = UV;

	return Res;
}

DEV int IntersectPlane(CRay R, bool OneSided, Vec3f Size, float* pT = NULL, Vec2f* pUV = NULL)
{
	Vec2f UV;

	int Res = IntersectPlane(R, OneSided, pT, &UV);

	if (Res <= 0)
		return 0;
	else
	{
		if (UV.x < -0.5f * Size.x || UV.x > 0.5f * Size.x || UV.y < -0.5f * Size.y || UV.y > 0.5f * Size.y)
			return 0;
	}

	if (pUV)
		*pUV = UV;

	return Res;
}

DEV int IntersectUnitDisk(CRay R, bool OneSided, float* pT = NULL, Vec2f* pUV = NULL)
{
	Vec2f UV;

	int Res = IntersectUnitPlane(R, OneSided, pT, &UV);

	if (Res <= 0)
		return 0;

	if (Res > 0)
	{
		if (UV.Length() > 0.5f)
			return 0;
	}

	if (pUV)
		*pUV = UV;

	return Res;
}

DEV int IntersectDisk(CRay R, bool OneSided, float Radius, float* pT = NULL, Vec2f* pUV = NULL)
{
	Vec2f UV;

	int Res = IntersectPlane(R, OneSided, pT, &UV);

	if (Res <= 0)
		return 0;

	if (Res > 0)
	{
		if (UV.Length() > Radius)
			return 0;
	}

	if (pUV)
		*pUV = UV;

	return Res;
}

DEV int IntersectUnitRing(CRay R, bool OneSided, float InnerRadius, float* pT = NULL, Vec2f* pUV = NULL)
{
	Vec2f UV;

	int Res = IntersectPlane(R, OneSided, pT, &UV);

	if (Res <= 0)
		return 0;

	if (Res > 0)
	{
		if (UV.Length() < InnerRadius || UV.Length() > 1.0f)
			return 0;
	}

	if (pUV)
		*pUV = UV;

	return Res;
}

DEV int IntersectRing(CRay R, bool OneSided, float InnerRadius, float OuterRadius, float* pT = NULL, Vec2f* pUV = NULL)
{
	Vec2f UV;

	int Res = IntersectPlane(R, OneSided, pT, &UV);

	if (Res <= 0)
		return 0;

	if (Res > 0)
	{
		if (UV.Length() < InnerRadius || UV.Length() > OuterRadius)
			return 0;
	}

	if (pUV)
		*pUV = UV;

	return Res;
}

DEV int IntersectUnitBox(CRay R, float* pNearT, float* pFarT)
{
	const Vec3f InvR		= Vec3f(1.0f, 1.0f, 1.0f) / R.m_D;
	const Vec3f BottomT		= InvR * (Vec3f(-0.5f) - R.m_O);
	const Vec3f TopT		= InvR * (Vec3f(0.5f) - R.m_O);
	const Vec3f MinT		= MinVec3f(TopT, BottomT);
	const Vec3f MaxT		= MaxVec3f(TopT, BottomT);
	const float LargestMinT = fmaxf(fmaxf(MinT.x, MinT.y), fmaxf(MinT.x, MinT.z));
	const float LargestMaxT = fminf(fminf(MaxT.x, MaxT.y), fminf(MaxT.x, MaxT.z));

	if (LargestMinT > 0.0f)
	{
		if (pNearT)
			*pNearT = LargestMinT;

		if (pFarT)
			*pFarT = LargestMaxT;
	}
	else
	{
		if (pNearT)
			*pNearT = 0.0f;

		if (pFarT)
			*pFarT = LargestMaxT;
	}

	if (*pNearT < R.m_MinT || *pNearT > R.m_MaxT)
		return 0;

	return 1;
}

DEV int IntersectBox(CRay R, Vec3f Min, Vec3f Max, float* pNearT, float* pFarT)
{
	const Vec3f InvR		= Vec3f(1.0f, 1.0f, 1.0f) / R.m_D;
	const Vec3f BottomT		= InvR * (Min - R.m_O);
	const Vec3f TopT		= InvR * (Max - R.m_O);
	const Vec3f MinT		= MinVec3f(TopT, BottomT);
	const Vec3f MaxT		= MaxVec3f(TopT, BottomT);
	const float LargestMinT = fmaxf(fmaxf(MinT.x, MinT.y), fmaxf(MinT.x, MinT.z));
	const float LargestMaxT = fminf(fminf(MaxT.x, MaxT.y), fminf(MaxT.x, MaxT.z));

	if (LargestMaxT < LargestMinT)
		return 0;

	if (LargestMinT > 0.0f)
	{
		if (pNearT)
			*pNearT = LargestMinT;

		if (pFarT)
			*pFarT = LargestMaxT;
	}
	else
	{
		if (pNearT)
			*pNearT = 0.0f;

		if (pFarT)
			*pFarT = LargestMaxT;
	}

	if (*pNearT < R.m_MinT || *pNearT > R.m_MaxT)
		return 0;

	return 1;
}

DEV bool IntersectBox(CRay R, Vec3f Size, float* pNearT, float* pFarT)
{
	return IntersectBox(R, -0.5f * Size, 0.5f * Size, pNearT, pFarT);
}

DEV bool InsideAABB(Vec3f P)
{
	if (P.x < gVolume.m_MinAABB[0] || P.x > gVolume.m_MaxAABB[0])
		return false;

	if (P.y < gVolume.m_MinAABB[1] || P.y > gVolume.m_MaxAABB[1])
		return false;

	if (P.z < gVolume.m_MinAABB[2] || P.z > gVolume.m_MaxAABB[2])
		return false;

	return true;
}

DEV bool IntersectSphere(CRay R, float Radius, float* pT)
{
	// http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection#Example_Code

    //Compute A, B and C coefficients
    float a = Dot(R.m_D, R.m_D);
	float b = 2 * Dot(R.m_D, R.m_O);
    float c = Dot(R.m_O, R.m_O) - (Radius * Radius);

    //Find discriminant
    const float disc = b * b - 4 * a * c;
    
    // if discriminant is negative there are no real roots, so return false, as ray misses sphere
    if (disc < 0)
        return 0;

    // compute q as described above
    float distSqrt = sqrtf(disc);
    float q;

    if (b < 0)
        q = (-b - distSqrt) / 2.0;
    else
        q = (-b + distSqrt) / 2.0;

    // compute t0 and t1
    float t0 = q / a;
    float t1 = c / q;

    // make sure t0 is smaller than t1
    if (t0 > t1)
    {
        // if t0 is bigger than t1 swap them around
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }


	if (t0 >= R.m_MinT && t0 < R.m_MaxT)
	{
		if (pT)
			*pT = t0;

        return 1;
	}
	else
	{
		if (t1 >= R.m_MinT && t1 < R.m_MaxT)
		{
			if (pT)
				*pT = t1;

			return 1;
		}
		else
		{
			return 0;
		}
	}
}

DEV bool IntersectUnitSphere(CRay R, float* pT)
{
    //Compute A, B and C coefficients
    float a = Dot(R.m_D, R.m_D);
	float b = 2 * Dot(R.m_D, R.m_O);
    float c = Dot(R.m_O, R.m_O)-1;

    //Find discriminant
    const float disc = b * b - 4 * a * c;
    
    // if discriminant is negative there are no real roots, so return 
    // false as ray misses sphere
    if (disc < 0)
        return 0;

    // compute q as described above
    float distSqrt = sqrtf(disc);
    float q;

    if (b < 0)
        q = (-b - distSqrt) / 2.0;
    else
        q = (-b + distSqrt) / 2.0;

    // compute t0 and t1
    float t0 = q / a;
    float t1 = c / q;

    // make sure t0 is smaller than t1
    if (t0 > t1)
    {
        // if t0 is bigger than t1 swap them around
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }


	if (t0 >= R.m_MinT && t0 < R.m_MaxT)
	{
		if (pT)
			*pT = t0;

        return 1;
	}
	else
	{
		if (t1 >= R.m_MinT && t1 < R.m_MaxT)
		{
			if (pT)
				*pT = t1;

			return 1;
		}
		else
		{
			return 0;
		}
	}
}

//DEV bool IntersectEllipsoid(CRay R, float Radius, float* pT)
//{
	// http://bjarni.us/ray-to-ellipsoid-intersection/
//}

DEV ColorXYZAf CumulativeMovingAverage(const ColorXYZAf& A, const ColorXYZAf& Ax, const int& N)
{
	 return A + ((Ax - A) / max((float)N, 1.0f));
}

DEV ColorXYZf CumulativeMovingAverage(const ColorXYZf& A, const ColorXYZf& Ax, const int& N)
{
	 return A + ((Ax - A) / max((float)N, 1.0f));
}