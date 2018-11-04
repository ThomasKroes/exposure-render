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

#include "geometry.h"
#include "disk.h"

// http://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node2.html
// http://mrl.nyu.edu/~dzorin/intro-graphics/lectures/lecture11/index.htm

namespace ExposureRender
{

HOST_DEVICE void IntersectCylinder(const Ray& R, const float& Radius, const float& Height, Intersection& Int)
{
	/*
	const float HalfHeight = 0.5f * Height;

	const float A = (R.D[0] * R.D[0]) + (R.D[1] * R.D[1]);
	const float B = 2.0f * (R.O[0]) * (R.D[0]) + 2.0f * (R.O[1]) * (R.D[1]);
	const float C = (R.O[0] * R.O[0]) + (R.O[1] * R.O[1]) - (Radius * Radius);
    const float D = (B * B) - 4.0f * A * C;
    
    if (D < 0)
		return;

	float T0 = (-B + sqrtf(D)) / 2.0f;
	float T1 = (-B - sqrtf(D)) / 2.0f;

	T0 /= A;
	T1 /= A;

    if (T0 > T1)
    {
        const float Temp = T0;
        T0 = T1;
        T1 = Temp;
    }

	Int.NearT	= T0;

	if (Int.NearT < R.MinT || Int.NearT > R.MaxT)
		return;

	Int.FarT	= T1;
	Int.P		= R(Int.NearT);

	if (Int.P[2] < -HalfHeight || Int.P[2] > HalfHeight)
	{
		Intersection Int1 = IntersectDisk(R, false, Radius, -HalfHeight);
		Intersection Int2 = IntersectDisk(R, false, Radius, HalfHeight);
		
		if (Int1.Valid && Int2.Valid)
		{
			if (Int1.NearT < Int2.NearT)
				return;

			if (Int2.NearT < Int1.NearT)
				return;
		}
		else
		{
			if (Int1.Valid)
				return;

			if (Int2.Valid)
				return;
		}
	}

	else
	{
		Int.N		= Vec3f(0.0f, 0.0f, 1.0f);
		Int.UV		= Vec2f(0.0f, 0.0f);
		Int.Valid	= true;
	}
	*/
}

HOST_DEVICE bool InsideCylinder(Vec3f P, float Radius, float Height)
{
	return sqrtf((P[0] * P[0]) + (P[2] * P[2])) < Radius && fabs(P[1]) < (0.5f * Height);
}

HOST_DEVICE void SampleCylinder(SurfaceSample& SS, Vec3f UVW, float Radius, float Height)
{
	/*
	int Side = floorf(UVW[2] * 3.0f);

	if (Side == 0 || Side == 1)
	{
		const Vec2f S = ConcentricSampleDisk(Vec2f(UVW[0], UVW[1]));
		
		SS.P[0]	= S[0];
		SS.P[1]	= S[1];
		SS.P[2]	= -0.5f * Height + Side * Height;
		SS.N	= Vec3f(0.0f, 0.0f, -1.0f);
	}

	if (Side == 2)
	{
		const float Theta = UVW[1] * TWO_PI_F;

		SS.P[0]	= cos(Theta);
		SS.P[1]	= sin(Theta);
		SS.P[2]	= -0.5f * Height + UVW[0] * Height;
		SS.N	= Vec3f(0.0f, 0.0f, 1.0f);
	}
	
	SS.UV = Vec2f(0.0f);
	*/

}

}
