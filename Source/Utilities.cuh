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

#include "Plane.cuh"
#include "Disk.cuh"
#include "Ring.cuh"
#include "Box.cuh"
#include "Sphere.cuh"
#include "Cylinder.cuh"

DEV Vec3f TransformVector(ErMatrix44 TM, Vec3f v)
{
  Vec3f r;

  float x = v[0], y = v[1], z = v[2];

  r[0] = TM.NN[0][0] * x + TM.NN[0][1] * y + TM.NN[0][2] * z;
  r[1] = TM.NN[1][0] * x + TM.NN[1][1] * y + TM.NN[1][2] * z;
  r[2] = TM.NN[2][0] * x + TM.NN[2][1] * y + TM.NN[2][2] * z;

  return r;
}

DEV Vec3f TransformPoint(ErMatrix44 TM, Vec3f pt)
{
	/*
	float x = pt.x, y = pt.y, z = pt.z;
    ptrans->x = m.m[0][0]*x + m.m[0][1]*y + m.m[0][2]*z + m.m[0][3];
    ptrans->y = m.m[1][0]*x + m.m[1][1]*y + m.m[1][2]*z + m.m[1][3];
    ptrans->z = m.m[2][0]*x + m.m[2][1]*y + m.m[2][2]*z + m.m[2][3];
    float w   = m.m[3][0]*x + m.m[3][1]*y + m.m[3][2]*z + m.m[3][3];
    if (w != 1.) *ptrans /= w;
	*/
    float x = pt[0], y = pt[1], z = pt[2];
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

DEV CRay TransformRay(CRay R, ErMatrix44 TM)
{
	CRay TR;

	Vec3f O		= TransformPoint(TM, R.m_O);
	Vec3f D		= TransformVector(TM, R.m_D);
	Vec3f MinP = R(R.m_MinT);
	Vec3f MaxP = R(R.m_MaxT);

	float MinT = Length(MinP - O);
	float MaxT = Length(MaxP - O);

	TR.m_O		= O;
	TR.m_D		= D;
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
	return make_float3(V[0], V[1], V[2]);
}

DEV float GetIntensity(Vec3f P)
{
	return (float)(USHRT_MAX * tex3D(gTexIntensity, (P[0] - gVolume.m_MinAABB[0]) * gVolume.m_InvSize[0], (P[1] - gVolume.m_MinAABB[1]) * gVolume.m_InvSize[1], (P[2] - gVolume.m_MinAABB[2]) * gVolume.m_InvSize[2]));
}

DEV float GetNormalizedIntensity(Vec3f P)
{
	return (GetIntensity(P) - gVolume.m_IntensityMin) * gVolume.m_IntensityInvRange;
}

DEV float GetOpacity(float NormalizedIntensity)
{
	return tex1D(gTexOpacity, NormalizedIntensity);
}

DEV bool Inside(ErClipper& C, Vec3f P)
{
	bool Inside = false;

	switch (C.ShapeType)
	{
		case 0:		
		{
			Inside = InsidePlane(P);
			break;
		}

		case 1:
		{
			Inside = InsideBox(P, ToVec3f(C.Size));
			break;
		}

		case 2:
		{
			Inside = InsideSphere(P, C.Radius);
			break;
		}

		case 3:
		{
			Inside = InsideCylinder(P, C.Radius, C.Size[1]);
			break;
		}
	}

	return C.Invert ? !Inside : Inside;
}

DEV bool Inside(Vec3f P)
{
	for (int i = 0; i < gClippers.NoClippers; i++)
	{
		ErClipper& C = gClippers.ClipperList[i];

		const Vec3f P2 = TransformPoint(C.InvTM, P);

		if (Inside(C, P2))
			return true;
	}

	return false;
}

DEV float GetOpacity(Vec3f P)
{
	const float Intensity = GetIntensity(P);
	
	const float NormalizedIntensity = (Intensity - gOpacityRange.m_Min) * gOpacityRange.m_InvRange;

	const float Opacity = GetOpacity(NormalizedIntensity);

	for (int i = 0; i < gClippers.NoClippers; i++)
	{
		ErClipper& C = gClippers.ClipperList[i];

		const Vec3f P2 = TransformPoint(C.InvTM, P);

		if (Inside(C, P2))
			return 0.0f;
	}

	return Opacity;
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

	Ints[0][0] = GetIntensity(Pts[0][0]);
	Ints[0][1] = GetIntensity(Pts[0][1]);
	Ints[1][0] = GetIntensity(Pts[1][0]);
	Ints[1][1] = GetIntensity(Pts[1][1]);
	Ints[2][0] = GetIntensity(Pts[2][0]);
	Ints[2][1] = GetIntensity(Pts[2][1]);

	Gradient[0] = (Ints[0][1] - Ints[0][0]) * gVolume.m_InvGradientDelta;
	Gradient[1] = (Ints[1][1] - Ints[1][0]) * gVolume.m_InvGradientDelta;
	Gradient[2] = (Ints[2][1] - Ints[2][0]) * gVolume.m_InvGradientDelta;

	return Normalize(Gradient);
}

DEV ColorXYZAf CumulativeMovingAverage(const ColorXYZAf& A, const ColorXYZAf& Ax, const int& N)
{
	 return A + ((Ax - A) / max((float)N, 1.0f));
}

DEV ColorXYZf CumulativeMovingAverage(const ColorXYZf& A, const ColorXYZf& Ax, const int& N)
{
	 return A + ((Ax - A) / max((float)N, 1.0f));
}

HOD ColorRGBuc ToneMap(ColorXYZAf XYZA)
{
	ColorRGBf RgbHdr;

	RgbHdr.FromXYZ(XYZA.GetX(), XYZA.GetY(), XYZA.GetZ());

	RgbHdr.SetR(Clamp(1.0f - expf(-(RgbHdr.GetR() * gCamera.m_InvExposure)), 0.0, 1.0f));
	RgbHdr.SetG(Clamp(1.0f - expf(-(RgbHdr.GetG() * gCamera.m_InvExposure)), 0.0, 1.0f));
	RgbHdr.SetB(Clamp(1.0f - expf(-(RgbHdr.GetB() * gCamera.m_InvExposure)), 0.0, 1.0f));
	
	ColorRGBuc Result;

	Result.SetR((unsigned char)Clamp((255.0f * powf(RgbHdr.GetR(), gCamera.m_InvGamma)), 0.0f, 255.0f));
	Result.SetG((unsigned char)Clamp((255.0f * powf(RgbHdr.GetG(), gCamera.m_InvGamma)), 0.0f, 255.0f));
	Result.SetB((unsigned char)Clamp((255.0f * powf(RgbHdr.GetB(), gCamera.m_InvGamma)), 0.0f, 255.0f));

	return Result;
}