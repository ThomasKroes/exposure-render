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

#include "color.h"
#include "ray.h"
#include "matrix.h"

using namespace std;

namespace ExposureRender
{

HOST_DEVICE inline Vec3f TransformVector(const Matrix44& TM, const Vec3f& V)
{
	Vec3f Vt;

	const float x = V[0], y = V[1], z = V[2];

	Vt[0] = TM.NN[0][0] * x + TM.NN[0][1] * y + TM.NN[0][2] * z;
	Vt[1] = TM.NN[1][0] * x + TM.NN[1][1] * y + TM.NN[1][2] * z;
	Vt[2] = TM.NN[2][0] * x + TM.NN[2][1] * y + TM.NN[2][2] * z;

	return Vt;
}

HOST_DEVICE inline Vec3f TransformPoint(const Matrix44& TM, const Vec3f& P)
{
	const float x = P[0], y = P[1], z = P[2];
    
	const float Px = TM.NN[0][0]*x + TM.NN[0][1]*y + TM.NN[0][2]*z + TM.NN[0][3];
    const float Py = TM.NN[1][0]*x + TM.NN[1][1]*y + TM.NN[1][2]*z + TM.NN[1][3];
    const float Pz = TM.NN[2][0]*x + TM.NN[2][1]*y + TM.NN[2][2]*z + TM.NN[2][3];
	
	return Vec3f(Px, Py, Pz);
}

HOST_DEVICE inline Ray TransformRay(const Matrix44& TM, const Ray& R)
{
	Ray Rt;

	Vec3f P		= TransformPoint(TM, R.O);
	Vec3f MinP	= TransformPoint(TM, R(R.MinT));
	Vec3f MaxP	= TransformPoint(TM, R(R.MaxT));

	Rt.O	= P;
	Rt.D	= Normalize(MaxP - Rt.O);
	Rt.MinT	= (MinP - Rt.O).Length();
	Rt.MaxT	= (MaxP - Rt.O).Length();

	return Rt;
}

HOST_DEVICE inline float SphericalTheta(const Vec3f& W)
{
	return acosf(Clamp(W[1], -1.0f, 1.0f));
}

HOST_DEVICE inline float SphericalPhi(const Vec3f& W)
{
	float p = atan2f(W[2], W[0]);
	return (p < 0.0f) ? p + 2.0f * PI_F : p;
}

HOST_DEVICE inline Vec2f SphericalToUV(const Vec3f& W)
{
	const Vec3f V = Normalize(W);
	return Vec2f(INV_TWO_PI_F * SphericalPhi(V), 1.0f - (INV_PI_F * SphericalTheta(V)));
}

HOST_DEVICE inline float Lerp(float t, float v1, float v2)
{
	return (1.f - t) * v1 + t * v2;
}

HOST_DEVICE inline void swap(int& a, int& b)
{
	int t = a; a = b; b = t;
}

HOST_DEVICE inline void swap(float& a, float& b)
{
	float t = a; a = b; b = t;
}

HOST_DEVICE inline void Swap(float* pF1, float* pF2)
{
	const float TempFloat = *pF1;

	*pF1 = *pF2;
	*pF2 = TempFloat;
}

HOST_DEVICE inline void Swap(float& F1, float& F2)
{
	const float TempFloat = F1;

	F1 = F2;
	F2 = TempFloat;
}

HOST_DEVICE inline void Swap(int* pI1, int* pI2)
{
	const int TempInt = *pI1;

	*pI1 = *pI2;
	*pI2 = TempInt;
}

HOST_DEVICE inline void Swap(int& I1, int& I2)
{
	const int TempInt = I1;

	I1 = I2;
	I2 = TempInt;

}

inline float RandomFloat(void)
{
	return (float)rand() / RAND_MAX;
}

}
