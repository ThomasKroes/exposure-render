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

#include "CudaUtilities.h"

#include "Geometry.cuh"
#include "Plane.cuh"

DEVICE Intersection IntersectUnitDisk(Ray R, bool OneSided)
{
	Intersection Int = IntersectUnitPlane(R, OneSided);

	if (Int.Valid && Int.UV.Length() > 0.5f)
		Int.Valid = false;

	return Int;
}

DEVICE bool IntersectDiskP(Ray R, bool OneSided, float Radius)
{
	Intersection Int = IntersectPlaneP(R, OneSided);

	if (Int.Valid && Int.UV.Length() > Radius)
		return false;

	return Int.Valid;
}

DEVICE Intersection IntersectDisk(Ray R, bool OneSided, float Radius)
{
	Intersection Int = IntersectPlane(R, OneSided);

	if (Int.Valid && Int.UV.Length() > Radius)
		Int.Valid = false;

	return Int;
}

DEVICE Intersection IntersectDisk(Ray R, bool OneSided, float Radius, float Offset)
{
	Intersection Int = IntersectPlane(R, OneSided, Offset);

	if (Int.Valid && Int.UV.Length() > Radius)
		Int.Valid = false;

	return Int;
}

DEVICE void SampleUnitDisk(SurfaceSample& SS, Vec3f UVW)
{
	float r = sqrtf(UVW[0]);
	float theta = 2.0f * PI_F * UVW[1];

	SS.P 	= Vec3f(r * cosf(theta), r * sinf(theta), 0.0f);
	SS.N 	= Vec3f(0.0f, 0.0f, 1.0f);
	SS.UV	= Vec2f(UVW[0], UVW[1]);
}

DEVICE void SampleDisk(SurfaceSample& SS, Vec3f UVW, float Radius)
{
	SampleUnitDisk(SS, UVW);
	
	SS.P *= Radius;
}