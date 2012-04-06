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
#include "Shape.cuh"
#include "MonteCarlo.cuh"
#include "Tracer.cuh"
										
namespace ExposureRender
{

DEVICE Vec3f TransformVector(const Matrix44& TM, const Vec3f& V)
{
	Vec3f Vt;

	const float x = V[0], y = V[1], z = V[2];

	Vt[0] = TM.NN[0][0] * x + TM.NN[0][1] * y + TM.NN[0][2] * z;
	Vt[1] = TM.NN[1][0] * x + TM.NN[1][1] * y + TM.NN[1][2] * z;
	Vt[2] = TM.NN[2][0] * x + TM.NN[2][1] * y + TM.NN[2][2] * z;

	return Vt;
}

DEVICE Vec3f TransformPoint(const Matrix44& TM, const Vec3f& P)
{
	const float x = P[0], y = P[1], z = P[2];
    
	const float Px = TM.NN[0][0]*x + TM.NN[0][1]*y + TM.NN[0][2]*z + TM.NN[0][3];
    const float Py = TM.NN[1][0]*x + TM.NN[1][1]*y + TM.NN[1][2]*z + TM.NN[1][3];
    const float Pz = TM.NN[2][0]*x + TM.NN[2][1]*y + TM.NN[2][2]*z + TM.NN[2][3];
	
	return Vec3f(Px, Py, Pz);
}

// Transform ray with transformation matrix
DEVICE Ray TransformRay(const Matrix44& TM, const Ray& R)
{
	Ray Rt;

	Vec3f P		= TransformPoint(TM, R.O);
	Vec3f MinP	= TransformPoint(TM, R(R.MinT));
	Vec3f MaxP	= TransformPoint(TM, R(R.MaxT));

	Rt.O	= P;
	Rt.D	= Normalize(MaxP - Rt.O);
	Rt.MinT	= (MinP - Rt.O).Length();
	Rt.MaxT	= (MaxP - Rt.O).Length();

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

DEVICE float GetIntensity(const Vec3f& P)
{
	return gpTracer->Volume.Get(P); 
}

DEVICE float GetNormalizedIntensity(const Vec3f& P)
{
	return (GetIntensity(P) - gpTracer->Volume.IntensityRange.Min) * gpTracer->Volume.IntensityRange.Inv;
}

DEVICE float GetOpacity(const float& NormalizedIntensity)
{
	return tex1D(gTexOpacity, NormalizedIntensity);
}

DEVICE bool Inside(ClippingObject& ClippingObject, Vec3f P)
{
	bool Inside = false;

	switch (ClippingObject.Shape.Type)
	{
		case 0:		
		{
			Inside = InsidePlane(P);
			break;
		}

		case 1:
		{
			Inside = InsideBox(P, ToVec3f(ClippingObject.Shape.Size));
			break;
		}

		case 2:
		{
			Inside = InsideSphere(P, ClippingObject.Shape.OuterRadius);
			break;
		}

		case 3:
		{
			Inside = InsideCylinder(P, ClippingObject.Shape.OuterRadius, ClippingObject.Shape.Size[1]);
			break;
		}
	}

	return ClippingObject.Invert ? !Inside : Inside;
}

DEVICE bool Inside(const Vec3f& P)
{
	for (int i = 0; i < gClippingObjects.Count; i++)
	{
		const Vec3f P2 = TransformPoint(gClippingObjects.List[i].Shape.InvTM, P);

		if (Inside(gClippingObjects.List[i], P2))
			return true;
	}

	return false;
}

DEVICE float GetOpacity(const Vec3f& P)
{
	const float Intensity = GetIntensity(P);
	
	const float NormalizedIntensity = (Intensity - gOpacityRange.Min) * gOpacityRange.Inv;

	const float Opacity = GetOpacity(NormalizedIntensity);

	for (int i = 0; i < gClippingObjects.Count; i++)
	{
		const Vec3f P2 = TransformPoint(gClippingObjects.List[i].Shape.InvTM, P);

		if (Inside(gClippingObjects.List[i], P2))
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
		{ GetIntensity(P + gpTracer->Volume.GradientDeltaX), GetIntensity(P - gpTracer->Volume.GradientDeltaX) },
		{ GetIntensity(P + gpTracer->Volume.GradientDeltaY), GetIntensity(P - gpTracer->Volume.GradientDeltaY) },
		{ GetIntensity(P + gpTracer->Volume.GradientDeltaZ), GetIntensity(P - gpTracer->Volume.GradientDeltaZ) }
	};

	return Vec3f(Intensity[0][1] - Intensity[0][0], Intensity[1][1] - Intensity[1][0], Intensity[2][1] - Intensity[2][0]);
}

DEVICE Vec3f GradientFD(Vec3f P)
{
	float Intensity[4] = 
	{
		GetIntensity(P),
		GetIntensity(P + gpTracer->Volume.GradientDeltaX),
		GetIntensity(P + gpTracer->Volume.GradientDeltaY),
		GetIntensity(P + gpTracer->Volume.GradientDeltaZ)
	};

    return Vec3f(Intensity[0] - Intensity[1], Intensity[0] - Intensity[2], Intensity[0] - Intensity[3]);
}

DEVICE Vec3f GradientFiltered(Vec3f P)
{
	Vec3f Offset(gpTracer->Volume.GradientDeltaX[0], gpTracer->Volume.GradientDeltaY[1], gpTracer->Volume.GradientDeltaZ[2]);

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
    
	return Lerp(G0, Lerp(L0, L1, 0.5), 0.75);
}

DEVICE Vec3f Gradient(Vec3f P)
{
	switch (gRenderSettings.Shading.GradientComputation)
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

	Pts[0][0] = P + gpTracer->Volume.GradientDeltaX;
	Pts[0][1] = P - gpTracer->Volume.GradientDeltaX;
	Pts[1][0] = P + gpTracer->Volume.GradientDeltaY;
	Pts[1][1] = P - gpTracer->Volume.GradientDeltaY;
	Pts[2][0] = P + gpTracer->Volume.GradientDeltaZ;
	Pts[2][1] = P - gpTracer->Volume.GradientDeltaZ;

	float D = 0.0f, Sum = 0.0f;

	for (int i = 0; i < 3; i++)
	{
		D = GetIntensity(Pts[i][1]) - GetIntensity(Pts[i][0]);
		D *= 0.5f / gpTracer->Volume.Spacing[i];
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
		// sample N-gon
        // FIXME: this could use concentric sampling
		float lensSides = 6.0f;
		float lensRotationRadians = 0.0f;
        float lensY = CS.LensUV[0] * lensSides;
        float side = (int)lensY;
        float offs = (float) lensY - side;
        float dist = (float) sqrtf(CS.LensUV[1]);
        float a0 = (float) (side * PI_F * 2.0f / lensSides + lensRotationRadians);
        float a1 = (float) ((side + 1.0f) * PI_F * 2.0f / lensSides + lensRotationRadians);
        float eyeX = (float) ((cos(a0) * (1.0f - offs) + cos(a1) * offs) * dist);
        float eyeY = (float) ((sin(a0) * (1.0f - offs) + sin(a1) * offs) * dist);
        eyeX *= gCamera.ApertureSize;
        eyeY *= gCamera.ApertureSize;

		const Vec2f LensUV(eyeX, eyeY);// = gCamera.ApertureSize * ConcentricSampleDisk(CS.LensUV);

		const Vec3f LI = ToVec3f(gCamera.U) * LensUV[0] + ToVec3f(gCamera.V) * LensUV[1];

		Rc.O += LI;
		Rc.D = Normalize(Rc.D * gCamera.FocalDistance - LI);
	}

	/*
	// sample N-gon
            // FIXME: this could use concentric sampling
            lensY *= lensSides;
            float side = (int) lensY;
            float offs = (float) lensY - side;
            float dist = (float) Math.sqrt(lensX);
            float a0 = (float) (side * Math.PI * 2.0f / lensSides + lensRotationRadians);
            float a1 = (float) ((side + 1.0f) * Math.PI * 2.0f / lensSides + lensRotationRadians);
            eyeX = (float) ((Math.cos(a0) * (1.0f - offs) + Math.cos(a1) * offs) * dist);
            eyeY = (float) ((Math.sin(a0) * (1.0f - offs) + Math.sin(a1) * offs) * dist);
            eyeX *= lensRadius;
            eyeY *= lensRadius;
			*/
}

DEVICE ColorXYZAf CumulativeMovingAverage(const ColorXYZAf& A, const ColorXYZAf& Ax, const int& N)
{
	return A + (Ax - A) / max((float)N, 1.0f);
}

DEVICE ColorXYZf CumulativeMovingAverage(const ColorXYZf& A, const ColorXYZf& Ax, const int& N)
{
	 return A + ((Ax - A) / max((float)N, 1.0f));
}

HOST_DEVICE float GlossinessExponent(const float& Glossiness)
{
	return 1000000.0f * powf(Glossiness, 7);
}

}
