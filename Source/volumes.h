/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or witDEVut modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software witDEVut specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT DEVLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT DEVLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) DEVWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Volume.h"

namespace ExposureRender
{

DEVICE float GetIntensity(const int& VolumeID, const Vec3f& P)
{
	return gpVolumes[VolumeID].Get(P);
}

DEVICE Vec3f GradientCD(const int& VolumeID, const Vec3f& P)
{
	const float Intensity[3][2] = 
	{
		{ GetIntensity(VolumeID, P + gpVolumes[VolumeID].GradientDeltaX), GetIntensity(VolumeID, P - gpVolumes[VolumeID].GradientDeltaX) },
		{ GetIntensity(VolumeID, P + gpVolumes[VolumeID].GradientDeltaY), GetIntensity(VolumeID, P - gpVolumes[VolumeID].GradientDeltaY) },
		{ GetIntensity(VolumeID, P + gpVolumes[VolumeID].GradientDeltaZ), GetIntensity(VolumeID, P - gpVolumes[VolumeID].GradientDeltaZ) }
	};

	return Vec3f(Intensity[0][1] - Intensity[0][0], Intensity[1][1] - Intensity[1][0], Intensity[2][1] - Intensity[2][0]);
}

DEVICE Vec3f GradientFD(const int& VolumeID, const Vec3f& P)
{
	const float Intensity[4] = 
	{
		GetIntensity(VolumeID, P),
		GetIntensity(VolumeID, P + gpVolumes[VolumeID].GradientDeltaX),
		GetIntensity(VolumeID, P + gpVolumes[VolumeID].GradientDeltaY),
		GetIntensity(VolumeID, P + gpVolumes[VolumeID].GradientDeltaZ)
	};

    return Vec3f(Intensity[0] - Intensity[1], Intensity[0] - Intensity[2], Intensity[0] - Intensity[3]);
}

DEVICE Vec3f GradientFiltered(const int& VolumeID, const Vec3f& P)
{
	Vec3f Offset(gpVolumes[VolumeID].GradientDeltaX[0], gpVolumes[VolumeID].GradientDeltaY[1], gpVolumes[VolumeID].GradientDeltaZ[2]);

    Vec3f G0 = GradientCD(VolumeID, P);
    Vec3f G1 = GradientCD(VolumeID, P + Vec3f(-Offset[0], -Offset[1], -Offset[2]));
    Vec3f G2 = GradientCD(VolumeID, P + Vec3f( Offset[0],  Offset[1],  Offset[2]));
    Vec3f G3 = GradientCD(VolumeID, P + Vec3f(-Offset[0],  Offset[1], -Offset[2]));
    Vec3f G4 = GradientCD(VolumeID, P + Vec3f( Offset[0], -Offset[1],  Offset[2]));
    Vec3f G5 = GradientCD(VolumeID, P + Vec3f(-Offset[0], -Offset[1],  Offset[2]));
    Vec3f G6 = GradientCD(VolumeID, P + Vec3f( Offset[0],  Offset[1], -Offset[2]));
    Vec3f G7 = GradientCD(VolumeID, P + Vec3f(-Offset[0],  Offset[1],  Offset[2]));
    Vec3f G8 = GradientCD(VolumeID, P + Vec3f( Offset[0], -Offset[1], -Offset[2]));
    
	Vec3f L0 = Lerp(Lerp(G1, G2, 0.5), Lerp(G3, G4, 0.5), 0.5);
    Vec3f L1 = Lerp(Lerp(G5, G6, 0.5), Lerp(G7, G8, 0.5), 0.5);
    
	return Lerp(G0, Lerp(L0, L1, 0.5), 0.75);
}

DEVICE Vec3f Gradient(const int& VolumeID, const Vec3f& P)
{
	switch (gpTracer->RenderSettings.Shading.GradientComputation)
	{
		case Enums::ForwardDifferences:	return GradientFD(VolumeID, P);
		case Enums::CentralDifferences:	return GradientCD(VolumeID, P);
		case Enums::Filtered:			return GradientFiltered(VolumeID, P);
	}

	return GradientFD(VolumeID, P);
}

DEVICE Vec3f NormalizedGradient(const int& VolumeID, const Vec3f& P)
{
	return Normalize(Gradient(VolumeID, P));
}

DEVICE float GradientMagnitude(const int& VolumeID, const Vec3f& P)
{
	Vec3f Pts[3][2];

	Pts[0][0] = P + gpVolumes[VolumeID].GradientDeltaX;
	Pts[0][1] = P - gpVolumes[VolumeID].GradientDeltaX;
	Pts[1][0] = P + gpVolumes[VolumeID].GradientDeltaY;
	Pts[1][1] = P - gpVolumes[VolumeID].GradientDeltaY;
	Pts[2][0] = P + gpVolumes[VolumeID].GradientDeltaZ;
	Pts[2][1] = P - gpVolumes[VolumeID].GradientDeltaZ;

	float D = 0.0f, Sum = 0.0f;

	for (int i = 0; i < 3; i++)
	{
		D = GetIntensity(VolumeID, Pts[i][1]) - GetIntensity(VolumeID, Pts[i][0]);
		D *= 0.5f / gpVolumes[VolumeID].Spacing[i];
		Sum += D * D;
	}

	return sqrtf(Sum);
}

}
