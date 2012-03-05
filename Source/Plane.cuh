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
#include "Sample.cuh"

#define INTERSECTION_EPSILON 0.0001f

DEV Intersection IntersectPlaneP(Ray R, bool OneSided, float Offset = 0.0f)
{
	Intersection Int;

	if (fabs(R.O[2] - R.D[2]) < INTERSECTION_EPSILON)
		return Int;

	Int.NearT = (Offset - R.O[2]) / R.D[2];
	
	if (Int.NearT < R.MinT || Int.NearT > R.MaxT)
		return Int;

	Int.UV	= Vec2f(Int.P[0], Int.P[1]);

	Int.Valid = true;

	return Int;
}

DEV Intersection IntersectPlane(Ray R, bool OneSided, float Offset = 0.0f)
{
	Intersection Int;

	if (fabs(R.O[2] - R.D[2]) < INTERSECTION_EPSILON)
		return Int;

	Int.NearT = (Offset - R.O[2]) / R.D[2];
	
	if (Int.NearT < R.MinT || Int.NearT > R.MaxT)
		return Int;

	Int.P 	= R(Int.NearT);
	Int.UV	= Vec2f(Int.P[0], Int.P[1]);
	Int.N	= Vec3f(0.0f, 0.0f, 1.0f);

	if (OneSided && R.D[2] <= 0.0f)
		Int.Front = false;

	Int.Valid = true;

	return Int;
}

DEV Intersection IntersectUnitPlane(Ray R, bool OneSided)
{
	Intersection Int = IntersectPlane(R, OneSided);

	if (Int.Valid && (Int.UV[0] < -0.5f || Int.UV[0] > 0.5f || Int.UV[1] < -0.5f || Int.UV[1] > 0.5f))
		Int.Valid = false;

	return Int;
}

DEV bool IntersectPlaneP(Ray R, bool OneSided, Vec2f Size)
{
	Intersection Int = IntersectPlaneP(R, OneSided);

	if (Int.Valid && (Int.UV[0] < -0.5f * Size[0] || Int.UV[0] > 0.5f * Size[0] || Int.UV[1] < -0.5f * Size[1] || Int.UV[1] > 0.5f * Size[1]))
		return false;

	return Int.Valid;
}

DEV Intersection IntersectPlane(Ray R, bool OneSided, Vec2f Size)
{
	Intersection Int = IntersectPlane(R, OneSided);

	if (Int.Valid && (Int.UV[0] < -0.5f * Size[0] || Int.UV[0] > 0.5f * Size[0] || Int.UV[1] < -0.5f * Size[1] || Int.UV[1] > 0.5f * Size[1]))
		Int.Valid = false;

	return Int;
}

DEV bool InsidePlane(Vec3f P)
{
	return P[2] > 0.0f;
}

HOD inline void SampleUnitPlane(SurfaceSample& SS, Vec3f UVW)
{
	SS.P 	= Vec3f(-0.5f + UVW[0], -0.5f + UVW[1], 0.0f);
	SS.N 	= Vec3f(0.0f, 0.0f, 1.0f);
	SS.UV	= Vec2f(UVW[0], UVW[1]);
}

HOD inline void SamplePlane(SurfaceSample& SS, Vec3f UVW, Vec2f Size)
{
	SampleUnitPlane(SS, UVW);

	SS.P *= Vec3f(Size[0], Size[1], 0.0f);
}