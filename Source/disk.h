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
#include "plane.h"

namespace ExposureRender
{

HOST_DEVICE bool IntersectDiskP(const Ray& R, const bool& OneSided, const float& Radius, Intersection& Int)
{
	IntersectPlaneP(R, OneSided, Int);

	if (Int.Valid && Int.UV.Length() > Radius)
		return false;

	return Int.Valid;
}

HOST_DEVICE void IntersectDisk(const Ray& R, const bool& OneSided, const float& Radius, Intersection& Int)
{
	IntersectPlane(R, OneSided, Int);

	if (Int.Valid && Int.UV.Length() > Radius)
		Int.Valid = false;

	const float Diameter = 2.0f * Radius;

	Int.UV /= Diameter;
	Int.UV += Vec2f(0.5f);
	Int.UV[0] = 1.0f - Int.UV[0];
}

HOST_DEVICE void IntersectDisk(const Ray& R, const bool& OneSided, const float& Radius, const float& Offset, Intersection& Int)
{
	IntersectPlane(R, OneSided, Offset, Int);

	if (Int.Valid && Int.UV.Length() > Radius)
		Int.Valid = false;
}

HOST_DEVICE void SampleUnitDisk(SurfaceSample& SS, const Vec3f& UVW)
{
	float r = sqrtf(UVW[0]);
	float theta = 2.0f * PI_F * UVW[1];

	SS.P 	= Vec3f(r * cosf(theta), r * sinf(theta), 0.0f);
	SS.N 	= Vec3f(0.0f, 0.0f, 1.0f);
	SS.UV	= Vec2f(UVW[0], UVW[1]);
}

HOST_DEVICE void SampleDisk(SurfaceSample& SS, const Vec3f& UVW, const float& Radius)
{
	SampleUnitDisk(SS, UVW);
	
	SS.P *= Radius;
}

}
