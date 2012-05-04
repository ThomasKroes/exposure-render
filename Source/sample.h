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
#include "color.h"
#include "rng.h"

namespace ExposureRender
{

HOST_DEVICE void Mutate1(float& X, CRNG& RNG, const float& S1 = 0.0009765625f, const float& S2 = 0.015625f)
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

HOST_DEVICE void Mutate2(Vec2f& V, CRNG& RNG, const float& S1 = 0.0009765625f, const float& S2 = 0.015625f)
{
	Mutate1(V[0], RNG, S1, S2);
	Mutate1(V[1], RNG, S1, S2);
}

HOST_DEVICE void Mutate3(Vec3f& V, CRNG& RNG, const float& S1 = 0.0009765625f, const float& S2 = 0.015625f)
{
	Mutate1(V[0], RNG, S1, S2);
	Mutate1(V[1], RNG, S1, S2);
	Mutate1(V[2], RNG, S1, S2);
}

class SurfaceSample
{
public:
	HOST_DEVICE SurfaceSample(void)
	{
		this->P		= Vec3f();
		this->N		= Vec3f(0.0f, 0.0f, 1.0f);
		this->UV	= Vec2f(0.0f);
	}

	HOST_DEVICE SurfaceSample& SurfaceSample::operator = (const SurfaceSample& Other)
	{
		this->P		= Other.P;
		this->N		= Other.N;
		this->UV	= Other.UV;

		return *this;
	}

	Vec3f	P;
	Vec3f	N;
	Vec2f	UV;
};

class LightSample
{
public:
	HOST_DEVICE LightSample(void)
	{
		this->SurfaceUVW = Vec3f();
	}

	HOST_DEVICE LightSample(CRNG& RNG)
	{
		this->LargeStep(RNG);
	}

	HOST_DEVICE LightSample& LightSample::operator=(const LightSample& Other)
	{
		this->SurfaceUVW = Other.SurfaceUVW;

		return *this;
	}

	HOST_DEVICE void LargeStep(CRNG& RNG)
	{
		this->SurfaceUVW = RNG.Get3();
	}

	HOST_DEVICE void Mutate(CRNG& RNG)
	{
		Mutate3(this->SurfaceUVW, RNG);
	}

	Vec3f 	SurfaceUVW;
};

class BrdfSample
{
public:
	HOST_DEVICE BrdfSample(void)
	{
		this->Component	= 0.0f;
		this->Dir 		= Vec2f(0.0f);
	}

	HOST_DEVICE BrdfSample(CRNG& RNG)
	{
		this->LargeStep(RNG);
	}

	HOST_DEVICE BrdfSample(const float& Component, const Vec2f& Dir)
	{
		this->Component	= Component;
		this->Dir 		= Dir;
	}

	HOST_DEVICE BrdfSample& BrdfSample::operator=(const BrdfSample& Other)
	{
		this->Component	= Other.Component;
		this->Dir 		= Other.Dir;

		return *this;
	}

	HOST_DEVICE void LargeStep(CRNG& RNG)
	{
		this->Component	= RNG.Get1();
		this->Dir		= RNG.Get2();
	}

	HOST_DEVICE void Mutate(CRNG& RNG)
	{
		Mutate1(this->Component, RNG);
		Mutate2(this->Dir, RNG);
	}

	float	Component;
	Vec2f	Dir;
};

class LightingSample
{
public:
	HOST_DEVICE LightingSample(void)
	{
		this->LightNum = 0.0f;
	}

	HOST_DEVICE LightingSample(CRNG& RNG)
	{
		this->LargeStep(RNG);
	}

	HOST_DEVICE LightingSample& LightingSample::operator=(const LightingSample& Other)
	{
		this->BrdfSample	= Other.BrdfSample;
		this->LightSample	= Other.LightSample;
		this->LightNum		= Other.LightNum;
		
		return *this;
	}

	HOST_DEVICE void LargeStep(CRNG& RNG)
	{
		this->BrdfSample.LargeStep(RNG);
		this->LightSample.LargeStep(RNG);
		this->LightNum = RNG.Get1();
	}

	HOST_DEVICE void Mutate(CRNG& RNG)
	{
		this->BrdfSample.Mutate(RNG);
		this->LightSample.Mutate(RNG);
		Mutate1(this->LightNum, RNG);
	}

	BrdfSample		BrdfSample;
	LightSample 	LightSample;
	float			LightNum;
};

class CameraSample
{
public:
	HOST_DEVICE CameraSample(void)
	{
		this->FilmUV	= Vec2f();
		this->LensUV	= Vec2f();
	}

	HOST_DEVICE CameraSample(CRNG& RNG)
	{
		this->LargeStep(RNG);
	}

	HOST_DEVICE CameraSample& CameraSample::operator=(const CameraSample& Other)
	{
		this->FilmUV	= Other.FilmUV;
		this->LensUV	= Other.LensUV;

		return *this;
	}

	HOST_DEVICE void LargeStep(CRNG& RNG)
	{
		this->FilmUV	= RNG.Get2();
		this->LensUV	= RNG.Get2();
	}

	HOST_DEVICE void Mutate(CRNG& RNG)
	{
		Mutate2(this->FilmUV, RNG, 0.001953125f, 0.0625f);
		Mutate2(this->LensUV, RNG);
	}

	Vec2f	FilmUV;
	Vec2f	LensUV;
};

class MetroSample
{
public:
	HOST_DEVICE MetroSample(void)
	{
	}

	HOST_DEVICE MetroSample(CRNG& RNG)
	{
		this->LargeStep(RNG);
	}

	HOST_DEVICE MetroSample& MetroSample::operator=(const MetroSample& Other)
	{
		this->LightingSample 	= Other.LightingSample;
		this->CameraSample		= Other.CameraSample;
		this->OldL				= Other.OldL;

		return *this;
	}

	HOST_DEVICE void LargeStep(CRNG& RNG)
	{
		this->LightingSample.LargeStep(RNG);
		this->CameraSample.LargeStep(RNG);
	}

	HOST_DEVICE MetroSample Mutate(CRNG& RNG)
	{
		MetroSample Result = *this;

		Result.LightingSample.Mutate(RNG);
		Result.CameraSample.Mutate(RNG);

		return Result;
	}

	LightingSample	LightingSample;
	CameraSample	CameraSample;
	ColorXYZAf		OldL;
};

}