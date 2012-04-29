/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or witDEVut modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software witDEVut specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT DEVLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT DEVLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) DEVWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "color.h"
#include "ray.h"
#include "matrix.h"

using namespace std;

namespace ExposureRender
{

class Intersection
{
public:
	HOST_DEVICE Intersection()
	{
		this->SetInvalid();
	}

	HOST_DEVICE void SetValid(float NearT, Vec3f P, Vec3f N, Vec2f UV = Vec2f(0.0f))
	{
		this->Valid		= true;
		this->NearT		= NearT;
		this->P			= P;
		this->N			= N;
		this->UV		= UV;
	}

	HOST_DEVICE void SetInvalid()
	{
		this->Valid		= false;
		this->Front		= true;
		this->NearT		= 0.0f;
		this->FarT		= FLT_MAX;
		this->P			= Vec3f();
		this->N			= Vec3f();
		this->UV		= Vec2f(0.0f);
	}

	HOST_DEVICE Intersection& Intersection::operator = (const Intersection& Other)
	{
		this->Valid			= Other.Valid;	
		this->Front			= Other.Front;
		this->NearT			= Other.NearT;
		this->FarT			= Other.FarT;
		this->P				= Other.P;
		this->N				= Other.N;
		this->UV			= Other.UV;

		return *this;
	}

	bool	Valid;
	bool	Front;
	float	NearT;
	float	FarT;
	Vec3f	P;
	Vec3f	N;
	Vec2f	UV;
};

}
