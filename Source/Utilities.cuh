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

DEV inline Vec3f ToVec3f(const float3& V)
{
	return Vec3f(V.x, V.y, V.z);
}

DEV inline float3 FromVec3f(const Vec3f& V)
{
	return make_float3(V.x, V.y, V.z);
}

DEV float GetIntensity(const Vec3f& P)
{
	return tex3D(gTexIntensity, (P.x - gVolume.m_MinAABB.x) * gVolume.m_InvSize.x, (P.y - gVolume.m_MinAABB.y) * gVolume.m_InvSize.y, (P.z - gVolume.m_MinAABB.z) * gVolume.m_InvSize.z);
}

DEV float GetNormalizedIntensity(const Vec3f& P)
{
	return (GetIntensity(P) - gVolume.m_IntensityMin) * gVolume.m_IntensityInvRange;
}

DEV float GetOpacity(const float& NormalizedIntensity)
{
	return tex1D(gTexOpacity, NormalizedIntensity);
}

DEV float GetOpacity(const Vec3f& P)
{
	for (int i = 0; i < gSlicing.m_NoSlices; i++)
	{
		Vec3f D = P - ToVec3f(gSlicing.m_Position[i]);

		if (Dot(D, ToVec3f(gSlicing.m_Normal[i])) < 0.0f)
			return 0.0f;
	}

	return GetOpacity(GetNormalizedIntensity(P));
}

DEV ColorXYZf GetDiffuse(const float& NormalizedIntensity)
{
	float4 Diffuse = tex1D(gTexDiffuse, NormalizedIntensity);
	return ColorXYZf(Diffuse.x, Diffuse.y, Diffuse.z);
}

DEV ColorXYZf GetSpecular(const float& NormalizedIntensity)
{
	float4 Specular = tex1D(gTexSpecular, NormalizedIntensity);
	return ColorXYZf(Specular.x, Specular.y, Specular.z);
}

DEV float GetGlossiness(const float& NormalizedIntensity)
{
	return tex1D(gTexGlossiness, NormalizedIntensity);
}

DEV float GetIOR(const float& NormalizedIntensity)
{
	return tex1D(gTexIOR, NormalizedIntensity);
}

DEV ColorXYZf GetEmission(const float& NormalizedIntensity)
{
	float4 Emission = tex1D(gTexEmission, NormalizedIntensity);
	return ColorXYZf(Emission.x, Emission.y, Emission.z);
}

DEV inline Vec3f NormalizedGradient(const Vec3f& P)
{
	Vec3f Gradient;

	Gradient.x = (GetIntensity(P + ToVec3f(gVolume.m_GradientDeltaX)) - GetIntensity(P - ToVec3f(gVolume.m_GradientDeltaX))) * gVolume.m_InvGradientDelta;
	Gradient.y = (GetIntensity(P + ToVec3f(gVolume.m_GradientDeltaY)) - GetIntensity(P - ToVec3f(gVolume.m_GradientDeltaY))) * gVolume.m_InvGradientDelta;
	Gradient.z = (GetIntensity(P + ToVec3f(gVolume.m_GradientDeltaZ)) - GetIntensity(P - ToVec3f(gVolume.m_GradientDeltaZ))) * gVolume.m_InvGradientDelta;

	return -Normalize(Gradient);
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

DEV bool IntersectUnitBox(CRay R, float* pNearT, float* pFarT)
{
	const Vec3f InvR		= Vec3f(1.0f, 1.0f, 1.0f) / R.m_D;
	const Vec3f BottomT		= InvR * (Vec3f(-0.5f) - R.m_O);
	const Vec3f TopT		= InvR * (Vec3f(0.5f) - R.m_O);
	const Vec3f MinT		= MinVec3f(TopT, BottomT);
	const Vec3f MaxT		= MaxVec3f(TopT, BottomT);
	const float LargestMinT = fmaxf(fmaxf(MinT.x, MinT.y), fmaxf(MinT.x, MinT.z));
	const float LargestMaxT = fminf(fminf(MaxT.x, MaxT.y), fminf(MaxT.x, MaxT.z));

	if (LargestMinT < R.m_MinT || LargestMinT > R.m_MaxT)
		return false;

	if (pNearT)
		*pNearT = LargestMinT;

	if (pFarT)
		*pFarT = LargestMaxT;

	return LargestMaxT > LargestMinT;
}

DEV bool IntersectBox(CRay R, Vec3f Min, Vec3f Max, float* pNearT, float* pFarT)
{
	const Vec3f InvR		= Vec3f(1.0f, 1.0f, 1.0f) / R.m_D;
	const Vec3f BottomT		= InvR * (Min - R.m_O);
	const Vec3f TopT		= InvR * (Max - R.m_O);
	const Vec3f MinT		= MinVec3f(TopT, BottomT);
	const Vec3f MaxT		= MaxVec3f(TopT, BottomT);
	const float LargestMinT = fmaxf(fmaxf(MinT.x, MinT.y), fmaxf(MinT.x, MinT.z));
	const float LargestMaxT = fminf(fminf(MaxT.x, MaxT.y), fminf(MaxT.x, MaxT.z));

	if (pNearT)
		*pNearT = LargestMinT;

	if (pFarT)
		*pFarT = LargestMaxT;

	if (LargestMaxT < LargestMinT || LargestMinT < R.m_MinT || LargestMinT > R.m_MaxT)
		return 0;
	else
		return 1;
}

DEV bool IntersectBox(CRay R, Vec3f Size, float* pNearT, float* pFarT)
{
	return IntersectBox(R, -0.5f * Size, 0.5f * Size, pNearT, pFarT);
}

DEV bool InsideAABB(Vec3f P)
{
	if (P.x < gVolume.m_MinAABB.x || P.x > gVolume.m_MaxAABB.x)
		return false;

	if (P.y < gVolume.m_MinAABB.y || P.y > gVolume.m_MaxAABB.y)
		return false;

	if (P.z < gVolume.m_MinAABB.z || P.z > gVolume.m_MaxAABB.z)
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
        return false;

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

        return true;
	}
	else
	{
		if (t1 >= R.m_MinT && t1 < R.m_MaxT)
		{
			if (pT)
				*pT = t1;

			return true;
		}
		else
		{
			return false;
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
        return false;

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

        return true;
	}
	else
	{
		if (t1 >= R.m_MinT && t1 < R.m_MaxT)
		{
			if (pT)
				*pT = t1;

			return true;
		}
		else
		{
			return false;
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