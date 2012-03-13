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

#define CUDA_SAFE_CALL_NO_SYNC(call) do {                                    \
    cudaError err = call;                                                    \
    if (cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString(err));                \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#define CUDA_SAFE_CALL(call) do {                                            \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if (cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString(err)) ;               \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

DEVICE Vec3f TransformVector(ErMatrix44 TM, Vec3f v)
{
	Vec3f r;

	float x = v[0], y = v[1], z = v[2];

	r[0] = TM.NN[0][0] * x + TM.NN[0][1] * y + TM.NN[0][2] * z;
	r[1] = TM.NN[1][0] * x + TM.NN[1][1] * y + TM.NN[1][2] * z;
	r[2] = TM.NN[2][0] * x + TM.NN[2][1] * y + TM.NN[2][2] * z;

	return r;
}

DEVICE Vec3f TransformPoint(ErMatrix44 TM, Vec3f pt)
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

// Transform ray with transformation matrix
DEVICE Ray TransformRay(Ray R, ErMatrix44& TM)
{
	Ray Rt;

	Vec3f P		= TransformPoint(TM, R.O);
	Vec3f MinP	= TransformPoint(TM, R(R.MinT));
	Vec3f MaxP	= TransformPoint(TM, R(R.MaxT));

	// Construct new ray
	Rt.O		= P;
	Rt.D		= Normalize(MaxP - Rt.O);
	Rt.MinT		= (MinP - Rt.O).Length();
	Rt.MaxT		= (MaxP - Rt.O).Length();

	return Rt;
}

HOST_DEVICE_NI Vec3f ToVec3f(float3 V)
{
	return Vec3f(V.x, V.y, V.z);
}

HOST_DEVICE_NI Vec3f ToVec3f(float V[3])
{
	return Vec3f(V[0], V[1], V[2]);
}

HOST_DEVICE_NI float3 FromVec3f(Vec3f V)
{
	return make_float3(V[0], V[1], V[2]);
}

HOST_DEVICE_NI ColorXYZf ToColorXYZf(float V[3])
{
	return ColorXYZf(V[0], V[1], V[2]);
}

DEVICE float GetIntensity(Vec3f P)
{
	return (float)(USHRT_MAX * tex3D(gTexIntensity, (P[0] - gVolume.MinAABB[0]) * gVolume.InvSize[0], (P[1] - gVolume.MinAABB[1]) * gVolume.InvSize[1], (P[2] - gVolume.MinAABB[2]) * gVolume.InvSize[2]));
}

DEVICE float GetNormalizedIntensity(Vec3f P)
{
	return (GetIntensity(P) - gVolume.IntensityRange.Min) * gVolume.IntensityRange.Inv;
}

DEVICE float GetOpacity(float NormalizedIntensity)
{
	return tex1D(gTexOpacity, NormalizedIntensity);
}

DEVICE bool Inside(ErClipper& C, Vec3f P)
{
	bool Inside = false;

	switch (C.Shape.Type)
	{
		case 0:		
		{
			Inside = InsidePlane(P);
			break;
		}

		case 1:
		{
			Inside = InsideBox(P, ToVec3f(C.Shape.Size));
			break;
		}

		case 2:
		{
			Inside = InsideSphere(P, C.Shape.OuterRadius);
			break;
		}

		case 3:
		{
			Inside = InsideCylinder(P, C.Shape.OuterRadius, C.Shape.Size[1]);
			break;
		}
	}

	return C.Invert ? !Inside : Inside;
}

DEVICE bool Inside(Vec3f P)
{
	for (int i = 0; i < gClippers.NoClippers; i++)
	{
		ErClipper& C = gClippers.ClipperList[i];

		const Vec3f P2 = TransformPoint(C.Shape.InvTM, P);

		if (Inside(C, P2))
			return true;
	}

	return false;
}

DEVICE float GetOpacity(Vec3f P)
{
	const float Intensity = GetIntensity(P);
	
	const float NormalizedIntensity = (Intensity - gOpacityRange.Min) * gOpacityRange.Inv;

	const float Opacity = GetOpacity(NormalizedIntensity);

	for (int i = 0; i < gClippers.NoClippers; i++)
	{
		ErClipper& C = gClippers.ClipperList[i];

		const Vec3f P2 = TransformPoint(C.Shape.InvTM, P);

		if (Inside(C, P2))
			return 0.0f;
	}

	return Opacity;
}

DEVICE ColorXYZf GetDiffuse(float Intensity)
{
	const float NormalizedIntensity = (Intensity - gDiffuseRange.Min) * gDiffuseRange.Inv;

	float4 Diffuse = tex1D(gTexDiffuse, NormalizedIntensity);
	return ColorXYZf(Diffuse.x, Diffuse.y, Diffuse.z);
}

DEVICE ColorXYZf GetSpecular(float Intensity)
{
	const float NormalizedIntensity = (Intensity - gSpecularRange.Min) * gSpecularRange.Inv;

	float4 Specular = tex1D(gTexSpecular, NormalizedIntensity);
	return ColorXYZf(Specular.x, Specular.y, Specular.z);
}

DEVICE float GetGlossiness(float Intensity)
{
	const float NormalizedIntensity = (Intensity - gGlossinessRange.Min) * gGlossinessRange.Inv;

	return tex1D(gTexGlossiness, NormalizedIntensity);
}

DEVICE ColorXYZf GetEmission(float Intensity)
{
	const float NormalizedIntensity = (Intensity - gEmissionRange.Min) * gEmissionRange.Inv;

	float4 Emission = tex1D(gTexEmission, NormalizedIntensity);
	return ColorXYZf(Emission.x, Emission.y, Emission.z);
}

DEVICE Vec3f GradientCD(Vec3f P)
{
	float Intensity[3][2] = 
	{
		{ GetIntensity(P + ToVec3f(gVolume.GradientDeltaX)), GetIntensity(P - ToVec3f(gVolume.GradientDeltaX)) },
		{ GetIntensity(P + ToVec3f(gVolume.GradientDeltaY)), GetIntensity(P - ToVec3f(gVolume.GradientDeltaY)) },
		{ GetIntensity(P + ToVec3f(gVolume.GradientDeltaZ)), GetIntensity(P - ToVec3f(gVolume.GradientDeltaZ)) }
	};

	return ToVec3f(gVolume.Spacing) * Vec3f(Intensity[0][1] - Intensity[0][0], Intensity[1][1] - Intensity[1][0], Intensity[2][1] - Intensity[2][0]);
}

DEVICE Vec3f GradientFD(Vec3f P)
{
	float Intensity[4] = 
	{
		GetIntensity(P),
		GetIntensity(P + ToVec3f(gVolume.GradientDeltaX)),
		GetIntensity(P + ToVec3f(gVolume.GradientDeltaY)),
		GetIntensity(P + ToVec3f(gVolume.GradientDeltaZ))
	};

    return ToVec3f(gVolume.Spacing) * Vec3f(Intensity[0] - Intensity[1], Intensity[0] - Intensity[2], Intensity[0] - Intensity[3]);
}

DEVICE Vec3f GradientFiltered(Vec3f P)
{
	Vec3f Offset(gVolume.GradientDeltaX[0], gVolume.GradientDeltaY[1], gVolume.GradientDeltaZ[2]);

    Vec3f G0 = GradientCD(P);
    Vec3f G1 = GradientCD(P + Vec3f(-Offset[0], -Offset[1], -Offset[2]));
    Vec3f G2 = GradientCD(P + Vec3f( Offset[0],  Offset[1],  Offset[2]));
    Vec3f G3 = GradientCD(P + Vec3f(-Offset[0],  Offset[1], -Offset[2]));
    Vec3f G4 = GradientCD(P + Vec3f( Offset[0], -Offset[1],  Offset[2]));
    Vec3f G5 = GradientCD(P + Vec3f(-Offset[0], -Offset[1],  Offset[2]));
    Vec3f G6 = GradientCD(P + Vec3f( Offset[0],  Offset[1], -Offset[2]));
    Vec3f G7 = GradientCD(P + Vec3f(-Offset[0],  Offset[1],  Offset[2]));
    Vec3f G8 = GradientCD(P + Vec3f( Offset[0], -Offset[1], -Offset[2]));
    
	Vec3f L0 = Lerp(Lerp(G1, G2, 0.5), Lerp(G3, G4, 0.5), 0.5);
    Vec3f L1 = Lerp(Lerp(G5, G6, 0.5), Lerp(G7, G8, 0.5), 0.5);
    
	return ToVec3f(gVolume.Spacing) * Lerp(G0, Lerp(L0, L1, 0.5), 0.75);
}

DEVICE Vec3f Gradient(Vec3f P)
{
	switch (gScattering.GradientComputation)
	{
		case 0:	return GradientFD(P);
		case 1:	return GradientCD(P);
		case 2:	return GradientFiltered(P);
	}

	return GradientFD(P);
}

DEVICE Vec3f NormalizedGradient(Vec3f P)
{
	return Normalize(Gradient(P));
}

DEVICE float GradientMagnitude(Vec3f P)
{
	Vec3f Pts[3][2];

	Pts[0][0] = P + ToVec3f(gVolume.GradientDeltaX);
	Pts[0][1] = P - ToVec3f(gVolume.GradientDeltaX);
	Pts[1][0] = P + ToVec3f(gVolume.GradientDeltaY);
	Pts[1][1] = P - ToVec3f(gVolume.GradientDeltaY);
	Pts[2][0] = P + ToVec3f(gVolume.GradientDeltaZ);
	Pts[2][1] = P - ToVec3f(gVolume.GradientDeltaZ);

	float D = 0.0f, Sum = 0.0f;

	for (int i = 0; i < 3; i++)
	{
		D = GetIntensity(Pts[i][1]) - GetIntensity(Pts[i][0]);
		D *= 0.5f / gVolume.Spacing[i];
		Sum += D * D;
	}

	return sqrtf(Sum);
}

DEVICE ColorRGBuc ToneMap(ColorXYZAf XYZA)
{
	ColorRGBf RgbHdr;

	RgbHdr.FromXYZ(XYZA.GetX(), XYZA.GetY(), XYZA.GetZ());

	RgbHdr.SetR(Clamp(1.0f - expf(-(RgbHdr.GetR() * gCamera.InvExposure)), 0.0, 1.0f));
	RgbHdr.SetG(Clamp(1.0f - expf(-(RgbHdr.GetG() * gCamera.InvExposure)), 0.0, 1.0f));
	RgbHdr.SetB(Clamp(1.0f - expf(-(RgbHdr.GetB() * gCamera.InvExposure)), 0.0, 1.0f));
	
	ColorRGBuc Result;

	Result.SetR((unsigned char)Clamp((255.0f * powf(RgbHdr.GetR(), gCamera.InvGamma)), 0.0f, 255.0f));
	Result.SetG((unsigned char)Clamp((255.0f * powf(RgbHdr.GetG(), gCamera.InvGamma)), 0.0f, 255.0f));
	Result.SetB((unsigned char)Clamp((255.0f * powf(RgbHdr.GetB(), gCamera.InvGamma)), 0.0f, 255.0f));

	return Result;
}

DEVICE float G(Vec3f P1, Vec3f N1, Vec3f P2, Vec3f N2)
{
	const Vec3f W = Normalize(P2 - P1);
	return (ClampedDot(W, N1) * ClampedDot(-W, N2)) / DistanceSquared(P1, P2);
}

DEVICE void SampleCamera(Ray& Rc, CameraSample& CS)
{
	Vec2f ScreenPoint;

	ScreenPoint[0] = gCamera.Screen[0][0] + (gCamera.InvScreen[0] * (float)(CS.FilmUV[0] * (float)gCamera.FilmWidth));
	ScreenPoint[1] = gCamera.Screen[1][0] + (gCamera.InvScreen[1] * (float)(CS.FilmUV[1] * (float)gCamera.FilmHeight));

	Rc.O	= ToVec3f(gCamera.Pos);
	Rc.D	= Normalize(ToVec3f(gCamera.N) + (ScreenPoint[0] * ToVec3f(gCamera.U)) - (ScreenPoint[1] * ToVec3f(gCamera.V)));
	Rc.MinT	= gCamera.ClipNear;
	Rc.MaxT	= gCamera.ClipFar;

	if (gCamera.ApertureSize != 0.0f)
	{
		const Vec2f LensUV = gCamera.ApertureSize * ConcentricSampleDisk(CS.LensUV);

		const Vec3f LI = ToVec3f(gCamera.U) * LensUV[0] + ToVec3f(gCamera.V) * LensUV[1];

		Rc.O += LI;
		Rc.D = Normalize(Rc.D * gCamera.FocalDistance - LI);
	}
}

DEVICE void SampleCamera(Ray& Rc, CameraSample& CS, const int& X, const int& Y)
{
	Vec2f FilmUV;

	FilmUV[0] = (float)X / (float)gCamera.FilmWidth;
	FilmUV[1] = (float)Y / (float)gCamera.FilmHeight;

	FilmUV[0] += CS.FilmUV[0] * (1.0f / (float)gCamera.FilmWidth);
	FilmUV[1] += CS.FilmUV[1] * (1.0f / (float)gCamera.FilmHeight);

	Vec2f ScreenPoint;

	ScreenPoint[0] = gCamera.Screen[0][0] + (gCamera.InvScreen[0] * (float)(FilmUV[0] * (float)gCamera.FilmWidth));
	ScreenPoint[1] = gCamera.Screen[1][0] + (gCamera.InvScreen[1] * (float)(FilmUV[1] * (float)gCamera.FilmHeight));

	Rc.O	= ToVec3f(gCamera.Pos);
	Rc.D	= Normalize(ToVec3f(gCamera.N) + (ScreenPoint[0] * ToVec3f(gCamera.U)) - (ScreenPoint[1] * ToVec3f(gCamera.V)));
	Rc.MinT	= gCamera.ClipNear;
	Rc.MaxT	= gCamera.ClipFar;

	if (gCamera.ApertureSize != 0.0f)
	{
		const Vec2f LensUV = gCamera.ApertureSize * ConcentricSampleDisk(CS.LensUV);

		const Vec3f LI = ToVec3f(gCamera.U) * LensUV[0] + ToVec3f(gCamera.V) * LensUV[1];

		Rc.O += LI;
		Rc.D = Normalize(Rc.D * gCamera.FocalDistance - LI);
	}
}

DEVICE ColorXYZAf CumulativeMovingAverage(const ColorXYZAf& A, const ColorXYZAf& Ax, const int& N)
{
	return A + (Ax - A) / max((float)N, 1.0f);
}

DEVICE ColorXYZf CumulativeMovingAverage(const ColorXYZf& A, const ColorXYZf& Ax, const int& N)
{
	 return A + ((Ax - A) / max((float)N, 1.0f));
}