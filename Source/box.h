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
