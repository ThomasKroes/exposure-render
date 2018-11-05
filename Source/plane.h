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

#include "intersection.h"
#include "sample.h"

namespace ExposureRender
{

HOST_DEVICE void IntersectPlaneP(const Ray& R, const bool& OneSided, Intersection& Int)
{
	if (fabs(R.O[2] - R.D[2]) < RAY_EPS)
		return;

	Int.NearT = (0.0f - R.O[2]) / R.D[2];
	
	if (Int.NearT < R.MinT || Int.NearT > R.MaxT)
		return;

	Int.UV		= Vec2f(Int.P[0], Int.P[1]);
	Int.Valid	= true;
}

HOST_DEVICE void IntersectPlane(const Ray& R, const bool& OneSided, Intersection& Int)
{
	if (fabs(R.O[2] - R.D[2]) < RAY_EPS)
		return;

	Int.NearT = (0.0f - R.O[2]) / R.D[2];
	
	if (Int.NearT < R.MinT || Int.NearT > R.MaxT)
		return;

	Int.P 	= R(Int.NearT);
	Int.UV	= Vec2f(Int.P[0], Int.P[1]);
	Int.N	= Vec3f(0.0f, 0.0f, 1.0f);

	if (OneSided && R.D[2] >= 0.0f)
	{
		Int.Front	= false;
		Int.N		= Vec3f(0.0f, 0.0f, -1.0f);
	}

	Int.Valid = true;
}

HOST_DEVICE bool IntersectPlaneP(const Ray& R, const bool& OneSided, const Vec2f& Size)
{
	Intersection Int;
	
	IntersectPlaneP(R, OneSided, Int);

	if (Int.Valid && (Int.UV[0] < -0.5f * Size[0] || Int.UV[0] > 0.5f * Size[0] || Int.UV[1] < -0.5f * Size[1] || Int.UV[1] > 0.5f * Size[1]))
		return false;
	
	return true;
}

HOST_DEVICE void IntersectPlane(const Ray& R, const bool& OneSided, const Vec2f& Size, Intersection& Int)
{
	IntersectPlane(R, OneSided, Int);

	if (Int.Valid && (Int.UV[0] < -0.5f * Size[0] || Int.UV[0] > 0.5f * Size[0] || Int.UV[1] < -0.5f * Size[1] || Int.UV[1] > 0.5f * Size[1]))
		Int.Valid = false;

	Int.UV[0] /= Size[0];
	Int.UV[1] /= Size[1];

	Int.UV += Vec2f(0.5f);
	Int.UV[0] = 1.0f - Int.UV[0];
}

HOST_DEVICE bool InsidePlane(Vec3f P)
{
	return P[2] > 0.0f;
}

HOST_DEVICE void SampleUnitPlane(SurfaceSample& SS, const Vec3f& UVW)
{
	SS.P 	= Vec3f(-0.5f + UVW[0], -0.5f + UVW[1], 0.0f);
	SS.N 	= Vec3f(0.0f, 0.0f, 1.0f);
	SS.UV	= Vec2f(UVW[0], UVW[1]);
}

HOST_DEVICE void SamplePlane(SurfaceSample& SS, const Vec3f& UVW, const Vec2f& Size)
{
	SampleUnitPlane(SS, UVW);

	SS.P *= Vec3f(Size[0], Size[1], 0.0f);
}

}
