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

HOST_DEVICE void IntersectUnitBox(const Ray& R, Intersection& Int)
{
	const Vec3f InvR		= Vec3f(1.0f, 1.0f, 1.0f) / R.D;
	const Vec3f BottomT		= InvR * (Vec3f(-0.5f) - R.O);
	const Vec3f TopT		= InvR * (Vec3f(0.5f) - R.O);
	const Vec3f MinT		= TopT.Min(BottomT);
	const Vec3f MaxT		= TopT.Max(BottomT);
	const float LargestMinT = fmaxf(fmaxf(MinT[0], MinT[1]), fmaxf(MinT[0], MinT[2]));
	const float LargestMaxT = fminf(fminf(MaxT[0], MaxT[1]), fminf(MaxT[0], MaxT[2]));

	if (LargestMaxT < LargestMinT)
		return;

	Int.NearT	= LargestMinT > 0.0f ? LargestMinT : 0.0f;
	Int.FarT	= LargestMaxT;

	if (Int.NearT < R.MinT || Int.NearT > R.MaxT)
		return;

	Int.Valid	= true;
	Int.P		= R(Int.NearT);
	Int.N		= Vec3f(0.0f, 0.0f, 1.0f);
	Int.UV		= Vec2f(0.0f, 0.0f);
}

HOST_DEVICE void IntersectBox(const Ray& R, const Vec3f& MinAABB, const Vec3f& MaxAABB, Intersection& Int)
{
	const Vec3f InvR		= Vec3f(1.0f, 1.0f, 1.0f) / R.D;
	const Vec3f BottomT		= InvR * (MinAABB - R.O);
	const Vec3f TopT		= InvR * (MaxAABB - R.O);
	const Vec3f MinT		= TopT.Min(BottomT);
	const Vec3f MaxT		= TopT.Max(BottomT);
	const float LargestMinT = fmaxf(fmaxf(MinT[0], MinT[1]), fmaxf(MinT[0], MinT[2]));
	const float LargestMaxT = fminf(fminf(MaxT[0], MaxT[1]), fminf(MaxT[0], MaxT[2]));

	if (LargestMaxT < LargestMinT)
		return;

	Int.NearT	= LargestMinT > 0.0f ? LargestMinT : 0.0f;
	Int.FarT	= LargestMaxT;

	if (Int.NearT < R.MinT || Int.NearT > R.MaxT)
		return;

	Int.Valid	= true;
	Int.P		= R(Int.NearT);
	Int.N		= Vec3f(0.0f);
	Int.UV		= Vec2f(0.0f, 0.0f);

	for (int i = 0; i < 3; i++)
	{
		if (Int.P[i] <= MinAABB[i] + 0.0001f)
			Int.N[i] = -1.0f;

		if (Int.P[i] >= MaxAABB[i] - 0.0001f)
			Int.N[i] = 1.0f;
	}
}

HOST_DEVICE void IntersectBox(const Ray& R, const Vec3f& Size, Intersection& Int)
{
	IntersectBox(R, -0.5f * Size, 0.5f * Size, Int);
}

HOST_DEVICE bool IntersectBoxP(const Ray& R, const Vec3f& MinAABB, const Vec3f& MaxAABB)
{
	const Vec3f InvR		= Vec3f(1.0f, 1.0f, 1.0f) / R.D;
	const Vec3f BottomT		= InvR * (MinAABB - R.O);
	const Vec3f TopT		= InvR * (MaxAABB - R.O);
	const Vec3f MinT		= TopT.Min(BottomT);
	const Vec3f MaxT		= TopT.Max(BottomT);
	const float LargestMinT = fmaxf(fmaxf(MinT[0], MinT[1]), fmaxf(MinT[0], MinT[2]));
	const float LargestMaxT = fminf(fminf(MaxT[0], MaxT[1]), fminf(MaxT[0], MaxT[2]));

	if (LargestMaxT < LargestMinT)
		return false;

	const float NearT = LargestMinT > 0.0f ? LargestMinT : 0.0f;

	if (NearT < R.MinT || NearT > R.MaxT)
		return false;

	return true;
}

HOST_DEVICE bool IntersectBoxP(Ray R, Vec3f Size)
{
	return IntersectBoxP(R, -0.5f * Size, 0.5f * Size);
}

HOST_DEVICE bool InsideBox(Vec3f P, Vec3f Size)
{
	const float HalfSize[3] = { 0.5f * Size[0], 0.5f * Size[1], 0.5f * Size[2] };
	return P[0] > -HalfSize[0] && P[0] < HalfSize[0] && P[1] > -HalfSize[1] && P[1] < HalfSize[1] && P[2] > -HalfSize[2] && P[2] < HalfSize[2];
}

HOST_DEVICE void SampleUnitBox(SurfaceSample& SS, Vec3f UVW)
{
	int Side = floorf(UVW[0] * 6.0f);

	switch (Side)
	{
		case 0:
		{
			SS.P[0] = 0.5f;
			SS.P[1] = -0.5f + UVW[2];
			SS.P[2] = -0.5f + UVW[1];
			SS.N	= Vec3f(1.0f, 0.0f, 0.0f);
			break;
		}

		case 1:
		{
			SS.P[0] = -0.5f;
			SS.P[1] = -0.5f + UVW[2];
			SS.P[2] = -0.5f + UVW[1];
			SS.N	= Vec3f(-1.0f, 0.0f, 0.0f);
			break;
		}

		case 2:
		{
			SS.P[0] = -0.5f + UVW[1];
			SS.P[1] = 0.5f;
			SS.P[2] = -0.5f + UVW[2];
			SS.N	= Vec3f(0.0f, 1.0f, 0.0f);
			break;
		}

		case 3:
		{
			SS.P[0] = -0.5f + UVW[1];
			SS.P[1] = -0.5f;
			SS.P[2] = -0.5f + UVW[2];
			SS.N	= Vec3f(0.0f, -1.0f, 0.0f);
			break;
		}

		case 4:
		{
			SS.P[0] = -0.5f + UVW[1];
			SS.P[1] = -0.5f + UVW[2];
			SS.P[2] = 0.5f;
			SS.N	= Vec3f(0.0f, 0.0f, 1.0f);
			break;
		}

		case 5:
		{
			SS.P[0] = -0.5f + UVW[1];
			SS.P[1] = -0.5f + UVW[2];
			SS.P[2] = -0.5f;
			SS.N	= Vec3f(0.0f, 0.0f, -1.0f);
			break;
		}
	}

	SS.UV = Vec2f(UVW[1], UVW[2]);
}

HOST_DEVICE void SampleBox(SurfaceSample& SS, Vec3f UVW, Vec3f Size)
{
	SampleUnitBox(SS, UVW);

	SS.P *= Size;
}

}
