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

namespace ExposureRender
{

// http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection#Example_Code

HOST_DEVICE bool IntersectSphereP(const Ray& R, const float& Radius)
{
	// Compute A, B and C coefficients
    float a = Dot(R.D, R.D);
	float b = 2 * Dot(R.D, R.O);
    float c = Dot(R.O, R.O) - (Radius * Radius);

    //Find discriminant
    const float disc = b * b - 4 * a * c;
    
    // if discriminant is negative there are no real roots, so return false, as ray misses sphere
    if (disc < 0)
        return false;

    // compute q as described above
    float distSqrt = sqrtf(disc);
    float q;

    if (b < 0)
        q = (-b - distSqrt) / 2.0f;
    else
        q = (-b + distSqrt) / 2.0f;

    // compute t0 and t1
    float t0 = q / a;
    float t1 = c / q;

    // make sure t0 is smaller than t1
    if (t0 > t1)
    {
        // if t0 is bigger than t1 swap them around
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }

	float NearT;

	if (t0 >= R.MinT && t0 < R.MaxT)
	{
		NearT = t0;
	}
	else
	{
		if (t1 >= R.MinT && t1 < R.MaxT)
			NearT = t1;
		else
			return false;
	}

	if (NearT < R.MinT || NearT > R.MaxT)
		return false;

	return true;
}

HOST_DEVICE void IntersectSphere(const Ray& R, const float& Radius, Intersection& Int)
{
	// Compute A, B and C coefficients
    float a = Dot(R.D, R.D);
	float b = 2 * Dot(R.D, R.O);
    float c = Dot(R.O, R.O) - (Radius * Radius);

    //Find discriminant
    const float disc = b * b - 4 * a * c;
    
    // if discriminant is negative there are no real roots, so return false, as ray misses sphere
    if (disc < 0)
        return;

    // compute q as described above
    float distSqrt = sqrtf(disc);
    float q;

    if (b < 0)
        q = (-b - distSqrt) / 2.0f;
    else
        q = (-b + distSqrt) / 2.0f;

    // compute t0 and t1
    float t0 = q / a;
    float t1 = c / q;

    // make sure t0 is smaller than t1
    if (t0 > t1)
    {
        // if t0 is bigger than t1 swap them around
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }


	if (t0 >= R.MinT && t0 < R.MaxT)
	{
		Int.NearT = t0;
	}
	else
	{
		if (t1 >= R.MinT && t1 < R.MaxT)
			Int.NearT = t1;
		else
			return;
	}

	if (Int.NearT < R.MinT || Int.NearT > R.MaxT)
		return;

	Int.Valid	= true;
	Int.P		= R(Int.NearT);
	Int.N		= Normalize(Int.P);
	Int.UV		= SphericalToUV(Int.P);
}

HOST_DEVICE void IntersectUnitSphere(const Ray& R, Intersection& Int)
{
	IntersectSphere(R, 1.0f, Int);
}

HOST_DEVICE bool InsideSphere(const Vec3f& P, const float& Radius)
{
	return Length(P) < Radius;
}

HOST_DEVICE void SampleUnitSphere(SurfaceSample& SS, const Vec3f& UVW)
{
	float z		= 1.0f - 2.0f * UVW[0];
	float r		= sqrtf(max(0.0f, 1.0f - z * z));
	float phi	= 2.0f * PI_F * UVW[1];
	float x		= r * cosf(phi);
	float y		= r * sinf(phi);

	SS.P	= Vec3f(x, y, z);
	SS.N	= SS.P;
	SS.UV	= Vec2f(SphericalTheta(SS.P), SphericalPhi(SS.P));
}

HOST_DEVICE void SampleSphere(SurfaceSample& SS, const Vec3f& UVW, const float& Radius)
{
	SampleUnitSphere(SS, UVW);

	SS.P *= Radius;
}

}
