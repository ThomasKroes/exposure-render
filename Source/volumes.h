/*
    Exposure Render: An interactive photo-realistic volume rendering framework
    Copyright (C) 2011 Thomas Kroes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "volume.h"

namespace ExposureRender
{

HOST_DEVICE_NI float GetIntensity(const int& VolumeID, const Vec3f& P)
{
	return gpVolumes[VolumeID](P);
}

HOST_DEVICE_NI Vec3f GradientCD(const int& VolumeID, const Vec3f& P)
{
	const float Intensity[3][2] = 
	{
		{ GetIntensity(VolumeID, P + gpVolumes[VolumeID].GradientDeltaX), GetIntensity(VolumeID, P - gpVolumes[VolumeID].GradientDeltaX) },
		{ GetIntensity(VolumeID, P + gpVolumes[VolumeID].GradientDeltaY), GetIntensity(VolumeID, P - gpVolumes[VolumeID].GradientDeltaY) },
		{ GetIntensity(VolumeID, P + gpVolumes[VolumeID].GradientDeltaZ), GetIntensity(VolumeID, P - gpVolumes[VolumeID].GradientDeltaZ) }
	};

	return Vec3f(Intensity[0][1] - Intensity[0][0], Intensity[1][1] - Intensity[1][0], Intensity[2][1] - Intensity[2][0]);
}

HOST_DEVICE_NI Vec3f GradientFD(const int& VolumeID, const Vec3f& P)
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

HOST_DEVICE_NI Vec3f GradientFiltered(const int& VolumeID, const Vec3f& P)
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

HOST_DEVICE_NI Vec3f Gradient(const int& VolumeID, const Vec3f& P)
{
	switch (gpTracer->RenderSettings.Shading.GradientComputation)
	{
		case Enums::ForwardDifferences:	return GradientFD(VolumeID, P);
		case Enums::CentralDifferences:	return GradientCD(VolumeID, P);
		case Enums::Filtered:			return GradientFiltered(VolumeID, P);
	}

	return GradientFD(VolumeID, P);
}

HOST_DEVICE_NI Vec3f NormalizedGradient(const int& VolumeID, const Vec3f& P)
{
	return Normalize(Gradient(VolumeID, P));
}

HOST_DEVICE_NI float GradientMagnitude(const int& VolumeID, const Vec3f& P)
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
