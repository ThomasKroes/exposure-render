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
#include "TransferFunction.cuh"
#include "Tracer.cuh"
#include "Volume.cuh"
										
namespace ExposureRender
{

HOST_DEVICE float GlossinessExponent(const float& Glossiness)
{
	return 1000000.0f * powf(Glossiness, 7);
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



/*
DEVICE bool Inside(ErClippingObject& ClippingObject, Vec3f P)
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
	for (int i = 0; i < gpTracer->ClippingObjects.Count; i++)
	{
		const Vec3f P2 = TransformPoint(gpTracer->ClippingObjects.List[i].Shape.InvTM, P);

		if (Inside(gpTracer->ClippingObjects.List[i], P2))
			return true;
	}

	return false;
}
*/

DEVICE float GetOpacity(const Vec3f& P)
{
	return EvaluateScalarTransferFunction1D(GetIntensity(P), gpTracer->Opacity1D);
}

DEVICE ColorXYZf GetDiffuse(const float& Intensity)
{
	return EvaluateColorTransferFunction1D(Intensity, gpTracer->Diffuse1D);
}

DEVICE ColorXYZf GetSpecular(const float& Intensity)
{
	return EvaluateColorTransferFunction1D(Intensity, gpTracer->Specular1D);
}

DEVICE float GetGlossiness(const float& Intensity)
{
	return GlossinessExponent(EvaluateScalarTransferFunction1D(Intensity, gpTracer->Glossiness1D));
}

DEVICE ColorXYZf GetEmission(const float& Intensity)
{
	return EvaluateColorTransferFunction1D(Intensity, gpTracer->Emission1D);
}

DEVICE Vec3f GradientCD(Vec3f P)
{
	float Intensity[3][2] = 
	{
		{ GetIntensity(P + gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaX), GetIntensity(P - gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaX) },
		{ GetIntensity(P + gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaY), GetIntensity(P - gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaY) },
		{ GetIntensity(P + gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaZ), GetIntensity(P - gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaZ) }
	};

	return Vec3f(Intensity[0][1] - Intensity[0][0], Intensity[1][1] - Intensity[1][0], Intensity[2][1] - Intensity[2][0]);
}

DEVICE Vec3f GradientFD(Vec3f P)
{
	float Intensity[4] = 
	{
		GetIntensity(P),
		GetIntensity(P + gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaX),
		GetIntensity(P + gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaY),
		GetIntensity(P + gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaZ)
	};

    return Vec3f(Intensity[0] - Intensity[1], Intensity[0] - Intensity[2], Intensity[0] - Intensity[3]);
}

DEVICE Vec3f GradientFiltered(Vec3f P)
{
	Vec3f Offset(gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaX[0], gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaY[1], gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaZ[2]);

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
	switch (gpTracer->RenderSettings.Shading.GradientComputation)
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

	Pts[0][0] = P + gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaX;
	Pts[0][1] = P - gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaX;
	Pts[1][0] = P + gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaY;
	Pts[1][1] = P - gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaY;
	Pts[2][0] = P + gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaZ;
	Pts[2][1] = P - gpVolumes->List[gpTracer->VolumeIDs[0]].GradientDeltaZ;

	float D = 0.0f, Sum = 0.0f;

	for (int i = 0; i < 3; i++)
	{
		D = GetIntensity(Pts[i][1]) - GetIntensity(Pts[i][0]);
		D *= 0.5f / gpVolumes->List[gpTracer->VolumeIDs[0]].Spacing[i];
		Sum += D * D;
	}

	return sqrtf(Sum);
}

DEVICE ColorRGBuc ToneMap(ColorXYZAf XYZA)
{
	ColorRGBf RgbHdr;

	RgbHdr.FromXYZ(XYZA.GetX(), XYZA.GetY(), XYZA.GetZ());

	RgbHdr.SetR(Clamp(1.0f - expf(-(RgbHdr.GetR() * gpTracer->Camera.InvExposure)), 0.0, 1.0f));
	RgbHdr.SetG(Clamp(1.0f - expf(-(RgbHdr.GetG() * gpTracer->Camera.InvExposure)), 0.0, 1.0f));
	RgbHdr.SetB(Clamp(1.0f - expf(-(RgbHdr.GetB() * gpTracer->Camera.InvExposure)), 0.0, 1.0f));
	
	ColorRGBuc Result;

	Result.SetR((unsigned char)Clamp((255.0f * powf(RgbHdr.GetR(), gpTracer->Camera.InvGamma)), 0.0f, 255.0f));
	Result.SetG((unsigned char)Clamp((255.0f * powf(RgbHdr.GetG(), gpTracer->Camera.InvGamma)), 0.0f, 255.0f));
	Result.SetB((unsigned char)Clamp((255.0f * powf(RgbHdr.GetB(), gpTracer->Camera.InvGamma)), 0.0f, 255.0f));

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

	ScreenPoint[0] = gpTracer->Camera.Screen[0][0] + (gpTracer->Camera.InvScreen[0] * (float)(CS.FilmUV[0] * (float)gpTracer->Camera.FilmWidth));
	ScreenPoint[1] = gpTracer->Camera.Screen[1][0] + (gpTracer->Camera.InvScreen[1] * (float)(CS.FilmUV[1] * (float)gpTracer->Camera.FilmHeight));

	Rc.O	= ToVec3f(gpTracer->Camera.Pos);
	Rc.D	= Normalize(ToVec3f(gpTracer->Camera.N) + (ScreenPoint[0] * ToVec3f(gpTracer->Camera.U)) - (ScreenPoint[1] * ToVec3f(gpTracer->Camera.V)));
	Rc.MinT	= gpTracer->Camera.ClipNear;
	Rc.MaxT	= gpTracer->Camera.ClipFar;

	if (gpTracer->Camera.ApertureSize != 0.0f)
	{
		const Vec2f LensUV = gpTracer->Camera.ApertureSize * ConcentricSampleDisk(CS.LensUV);

		const Vec3f LI = ToVec3f(gpTracer->Camera.U) * LensUV[0] + ToVec3f(gpTracer->Camera.V) * LensUV[1];

		Rc.O += LI;
		Rc.D = Normalize(Rc.D * gpTracer->Camera.FocalDistance - LI);
	}
}

DEVICE void SampleCamera(Ray& Rc, CameraSample& CS, const int& X, const int& Y)
{
	Vec2f FilmUV;

	FilmUV[0] = (float)X / (float)gpTracer->Camera.FilmWidth;
	FilmUV[1] = (float)Y / (float)gpTracer->Camera.FilmHeight;

	FilmUV[0] += CS.FilmUV[0] * (1.0f / (float)gpTracer->Camera.FilmWidth);
	FilmUV[1] += CS.FilmUV[1] * (1.0f / (float)gpTracer->Camera.FilmHeight);

	Vec2f ScreenPoint;

	ScreenPoint[0] = gpTracer->Camera.Screen[0][0] + (gpTracer->Camera.InvScreen[0] * (float)(FilmUV[0] * (float)gpTracer->Camera.FilmWidth));
	ScreenPoint[1] = gpTracer->Camera.Screen[1][0] + (gpTracer->Camera.InvScreen[1] * (float)(FilmUV[1] * (float)gpTracer->Camera.FilmHeight));

	Rc.O	= ToVec3f(gpTracer->Camera.Pos);
	Rc.D	= Normalize(ToVec3f(gpTracer->Camera.N) + (ScreenPoint[0] * ToVec3f(gpTracer->Camera.U)) - (ScreenPoint[1] * ToVec3f(gpTracer->Camera.V)));
	Rc.MinT	= gpTracer->Camera.ClipNear;
	Rc.MaxT	= gpTracer->Camera.ClipFar;

	if (gpTracer->Camera.ApertureSize != 0.0f)
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
        eyeX *= gpTracer->Camera.ApertureSize;
        eyeY *= gpTracer->Camera.ApertureSize;

		const Vec2f LensUV(eyeX, eyeY);// = gpTracer->Camera.ApertureSize * ConcentricSampleDisk(CS.LensUV);

		const Vec3f LI = ToVec3f(gpTracer->Camera.U) * LensUV[0] + ToVec3f(gpTracer->Camera.V) * LensUV[1];

		Rc.O += LI;
		Rc.D = Normalize(Rc.D * gpTracer->Camera.FocalDistance - LI);
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

}
