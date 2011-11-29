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

DEV float GetIntensity(const Vec3f& P)
{
	return ((float)SHRT_MAX * tex3D(gTexDensity, (P.x - gVolumeInfo.m_MinAABB.x) * gVolumeInfo.m_InvExtent.x, (P.y - gVolumeInfo.m_MinAABB.y) * gVolumeInfo.m_InvExtent.y, (P.z - gVolumeInfo.m_MinAABB.z) * gVolumeInfo.m_InvExtent.z));
}

DEV float GetNormalizedIntensity(const Vec3f& P)
{
	return (GetIntensity(P) - gVolumeInfo.m_IntensityMin) * gVolumeInfo.m_IntensityInvRange;
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

	Gradient.x = (GetIntensity(P + ToVec3f(gVolumeInfo.m_GradientDeltaX)) - GetIntensity(P - ToVec3f(gVolumeInfo.m_GradientDeltaX)));// * gVolumeInfo.m_InvGradientDelta;
	Gradient.y = (GetIntensity(P + ToVec3f(gVolumeInfo.m_GradientDeltaY)) - GetIntensity(P - ToVec3f(gVolumeInfo.m_GradientDeltaY)));// * gVolumeInfo.m_InvGradientDelta;
	Gradient.z = (GetIntensity(P + ToVec3f(gVolumeInfo.m_GradientDeltaZ)) - GetIntensity(P - ToVec3f(gVolumeInfo.m_GradientDeltaZ)));// * gVolumeInfo.m_InvGradientDelta;

	return -Normalize(Gradient);
}

DEV float GradientMagnitude(const Vec3f& P)
{
	return ((float)SHRT_MAX * tex3D(gTexGradientMagnitude, P.x * gVolumeInfo.m_InvMaxAABB.x, P.y * gVolumeInfo.m_InvMaxAABB.y, P.z * gVolumeInfo.m_InvMaxAABB.z));
}

DEV bool IntersectPlane(const CRay& R, bool OneSided, Vec3f P, Vec3f N, Vec3f U, Vec3f V, Vec2f Luv, float* pT = NULL, Vec2f* pUV = NULL)
{
	const float DotN = Dot(R.m_D, N);

	if (OneSided && DotN >= 0.0f)
		return false;

	const float T = (Dot((P - R.m_O).Length(), N)) / DotN;

	if (T < R.m_MinT || T > R.m_MaxT)
		return false;

	const Vec3f Pl = R(T);

	const Vec3f Wl = Pl - P;

	const Vec2f UV = Vec2f(Dot(Wl, U), Dot(Wl, V));

	const float HalfLu = 0.5f * Luv.x;
	const float HalfLv = 0.5f * Luv.y;

	if (UV.x > HalfLu || UV.x < -HalfLu || UV.y > HalfLv || UV.y < -HalfLv)
		return false;

	if (pUV)
		*pUV = UV;

	return true;
}

DEV bool IntersectBox(const CRay& R, float* pNearT, float* pFarT)
{
	const Vec3f InvR		= Vec3f(1.0f, 1.0f, 1.0f) / R.m_D;
	const Vec3f BottomT		= InvR * (Vec3f(gVolumeInfo.m_MinAABB.x, gVolumeInfo.m_MinAABB.y, gVolumeInfo.m_MinAABB.z) - R.m_O);
	const Vec3f TopT		= InvR * (Vec3f(gVolumeInfo.m_MaxAABB.x, gVolumeInfo.m_MaxAABB.y, gVolumeInfo.m_MaxAABB.z) - R.m_O);
	const Vec3f MinT		= MinVec3f(TopT, BottomT);
	const Vec3f MaxT		= MaxVec3f(TopT, BottomT);
	const float LargestMinT = fmaxf(fmaxf(MinT.x, MinT.y), fmaxf(MinT.x, MinT.z));
	const float LargestMaxT = fminf(fminf(MaxT.x, MaxT.y), fminf(MaxT.x, MaxT.z));

	*pNearT = LargestMinT;
	*pFarT	= LargestMaxT;

	return LargestMaxT > LargestMinT;
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

	return false;
}

DEV bool IntersectEllipsoid(CRay R, float Radius, float* pT)
{
	// http://bjarni.us/ray-to-ellipsoid-intersection/
}

DEV ColorXYZAf CumulativeMovingAverage(const ColorXYZAf& A, const ColorXYZAf& Ax, const int& N)
{
	 return A + ((Ax - A) / max((float)N, 1.0f));
}

DEV ColorXYZf CumulativeMovingAverage(const ColorXYZf& A, const ColorXYZf& Ax, const int& N)
{
	 return A + ((Ax - A) / max((float)N, 1.0f));
}