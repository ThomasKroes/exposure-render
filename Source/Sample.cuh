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

#include "Geometry.cuh"
#include "Color.h"
#include "RNG.cuh"

namespace ExposureRender
{

DEVICE void Mutate1(float& X, CRNG& RNG, const float& S1 = 0.0009765625f, const float& S2 = 0.015625f)
{
	float dx = S2 * exp(-log(S2 / S1) * RNG.Get1());

	if (RNG.Get1() < 0.5f)
	{
		float x1 = X + dx;
		X = (x1 > 1) ? x1 - 1 : x1;
	}
	else
	{
		float x1 = X - dx;
		X = (x1 < 0) ? x1 + 1 : x1;
	}
}

DEVICE void Mutate2(Vec2f& V, CRNG& RNG, const float& S1 = 0.0009765625f, const float& S2 = 0.015625f)
{
	Mutate1(V[0], RNG, S1, S2);
	Mutate1(V[1], RNG, S1, S2);
}

DEVICE void Mutate3(Vec3f& V, CRNG& RNG, const float& S1 = 0.0009765625f, const float& S2 = 0.015625f)
{
	Mutate1(V[0], RNG, S1, S2);
	Mutate1(V[1], RNG, S1, S2);
	Mutate1(V[2], RNG, S1, S2);
}

struct SurfaceSample
{
	Vec3f		P;
	Vec3f		N;
	Vec2f		UV;

	DEVICE SurfaceSample(void)
	{
		this->P		= Vec3f(0.0f);
		this->N		= Vec3f(0.0f, 0.0f, 1.0f);
		this->UV	= Vec2f(0.0f);
	}

	DEVICE SurfaceSample& SurfaceSample::operator = (const SurfaceSample& Other)
	{
		this->P		= Other.P;
		this->N		= Other.N;
		this->UV	= Other.UV;

		return *this;
	}
};

struct LightSample
{
	Vec3f 	SurfaceUVW;

	DEVICE LightSample(void)
	{
		this->SurfaceUVW = Vec3f(0.0f);
	}

	DEVICE LightSample(CRNG& RNG)
	{
		this->LargeStep(RNG);
	}

	DEVICE LightSample& LightSample::operator=(const LightSample& Other)
	{
		this->SurfaceUVW = Other.SurfaceUVW;

		return *this;
	}

	DEVICE void LargeStep(CRNG& RNG)
	{
		this->SurfaceUVW = RNG.Get3();
	}

	DEVICE void Mutate(CRNG& RNG)
	{
		Mutate3(this->SurfaceUVW, RNG);
	}
};

struct BrdfSample
{
	float	Component;
	Vec2f	Dir;

	DEVICE BrdfSample(void)
	{
		this->Component	= 0.0f;
		this->Dir 		= Vec2f(0.0f);
	}

	DEVICE BrdfSample(CRNG& RNG)
	{
		this->LargeStep(RNG);
	}

	DEVICE BrdfSample(const float& Component, const Vec2f& Dir)
	{
		this->Component	= Component;
		this->Dir 		= Dir;
	}

	DEVICE BrdfSample& BrdfSample::operator=(const BrdfSample& Other)
	{
		this->Component	= Other.Component;
		this->Dir 		= Other.Dir;

		return *this;
	}

	DEVICE void LargeStep(CRNG& RNG)
	{
		this->Component	= RNG.Get1();
		this->Dir		= RNG.Get2();
	}

	DEVICE void Mutate(CRNG& RNG)
	{
		Mutate1(this->Component, RNG);
		Mutate2(this->Dir, RNG);
	}
};

struct LightingSample
{
	BrdfSample		BrdfSample;
	LightSample 	LightSample;
	float			LightNum;

	DEVICE LightingSample(void)
	{
		this->LightNum = 0.0f;
	}

	DEVICE LightingSample(CRNG& RNG)
	{
		this->LargeStep(RNG);
	}

	DEVICE LightingSample& LightingSample::operator=(const LightingSample& Other)
	{
		this->BrdfSample	= Other.BrdfSample;
		this->LightSample	= Other.LightSample;
		this->LightNum		= Other.LightNum;
		
		return *this;
	}

	DEVICE void LargeStep(CRNG& RNG)
	{
		this->BrdfSample.LargeStep(RNG);
		this->LightSample.LargeStep(RNG);
		this->LightNum = RNG.Get1();
	}

	DEVICE void Mutate(CRNG& RNG)
	{
		this->BrdfSample.Mutate(RNG);
		this->LightSample.Mutate(RNG);
		Mutate1(this->LightNum, RNG);
	}
};

struct CameraSample
{
	Vec2f	FilmUV;
	Vec2f	LensUV;

	DEVICE CameraSample(void)
	{
		this->FilmUV	= Vec2f();
		this->LensUV	= Vec2f();
	}

	DEVICE CameraSample(CRNG& RNG)
	{
		this->LargeStep(RNG);
	}

	DEVICE CameraSample& CameraSample::operator=(const CameraSample& Other)
	{
		this->FilmUV	= Other.FilmUV;
		this->LensUV	= Other.LensUV;

		return *this;
	}

	DEVICE void LargeStep(CRNG& RNG)
	{
		this->FilmUV	= RNG.Get2();
		this->LensUV	= RNG.Get2();
	}

	DEVICE void Mutate(CRNG& RNG)
	{
		Mutate2(this->FilmUV, RNG, 0.001953125f, 0.0625f);
		Mutate2(this->LensUV, RNG);
	}
};

struct MetroSample
{
	LightingSample	LightingSample;
	CameraSample	CameraSample;
	ColorXYZAf		OldL;

	DEVICE MetroSample(void)
	{
	}

	DEVICE MetroSample(CRNG& RNG)
	{
		this->LargeStep(RNG);
	}

	DEVICE MetroSample& MetroSample::operator=(const MetroSample& Other)
	{
		this->LightingSample 	= Other.LightingSample;
		this->CameraSample		= Other.CameraSample;
		this->OldL				= Other.OldL;

		return *this;
	}

	DEVICE void LargeStep(CRNG& RNG)
	{
		this->LightingSample.LargeStep(RNG);
		this->CameraSample.LargeStep(RNG);
	}

	DEVICE MetroSample Mutate(CRNG& RNG)
	{
		MetroSample Result = *this;

		Result.LightingSample.Mutate(RNG);
		Result.CameraSample.Mutate(RNG);

		return Result;
	}
};

}