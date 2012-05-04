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

#include "geometry.h"
#include "disk.h"

namespace ExposureRender
{

HOST_DEVICE void IntersectUnitRing(const Ray& R, const bool& OneSided, const float& InnerRadius, Intersection& Int)
{
	IntersectPlane(R, OneSided, Int);

	if (Int.Valid && (Int.UV.Length() < InnerRadius || Int.UV.Length() >= 1.0f))
		Int.Valid = false;
}

HOST_DEVICE bool IntersectRingP(const Ray& R, const bool& OneSided, const float& InnerRadius, const float& OuterRadius, Intersection& Int)
{
	IntersectPlaneP(R, OneSided, Int);

	if (Int.Valid && (Int.UV.Length() < InnerRadius || Int.UV.Length() > OuterRadius))
		return false;

	return Int.Valid;
}

HOST_DEVICE void IntersectRing(const Ray& R, const bool& OneSided, const float& InnerRadius, const float& OuterRadius, Intersection& Int)
{
	IntersectPlane(R, OneSided, Int);

	if (Int.Valid && (Int.UV.Length() < InnerRadius || Int.UV.Length() > OuterRadius))
		Int.Valid = false;

	const float Diameter = 2.0f * OuterRadius;

	Int.UV /= Diameter;
	Int.UV += Vec2f(0.5f);
	Int.UV[0] = 1.0f - Int.UV[0]; 
}

HOST_DEVICE void SampleUnitRing(SurfaceSample& SS, const Vec3f& UVW, const float& InnerRadius)
{
	float r = InnerRadius + (1.0f - InnerRadius) * sqrtf(UVW[0]);
	float theta = 2.0f * PI_F * UVW[1];

	SS.P 	= Vec3f(r * cosf(theta), r * sinf(theta), 0.0f);
	SS.N 	= Vec3f(0.0f, 0.0f, 1.0f);
	SS.UV	= Vec2f(SS.P[0], SS.P[1]);
}

HOST_DEVICE void SampleRing(SurfaceSample& SS, const Vec3f& UVW, const float& InnerRadius, const float& OuterRadius)
{
	SampleUnitRing(SS, UVW, InnerRadius / OuterRadius);

	SS.P *= OuterRadius;
	SS.UV	= Vec2f(SS.P[0], SS.P[1]);
}

}
