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

class ScatterEvent
{
public:
	enum SampleType
	{
		Volume,
		Light,
		Object,
		SlicePlane
	};

	HOST_DEVICE ScatterEvent()
	{
	}

	HOST_DEVICE ScatterEvent(const SampleType& Type)
	{
		this->Type = Type;
		this->SetInvalid();
	}

	HOST_DEVICE void SetValid(float T, Vec3f P, Vec3f N, Vec3f Wo, ColorXYZf Le, Vec2f UV = Vec2f(0.0f))
	{
		this->Valid		= true;
		this->T			= T;
		this->P			= P;
		this->N			= N;
		this->Wo		= Wo;
		this->Le		= Le;
		this->UV		= UV;
	}

	HOST_DEVICE void SetInvalid()
	{
		this->Valid		= false;
		this->T			= 0.0f;
		this->P			= Vec3f();
		this->N			= Vec3f();
		this->Wo		= Vec3f();
		this->Le		= ColorXYZf(0.0f);
		this->UV		= Vec2f(0.0f);
	}

	HOST_DEVICE ScatterEvent& ScatterEvent::operator = (const ScatterEvent& Other)
	{
		this->Type			= Other.Type;
		this->Valid			= Other.Valid;	
		this->T				= Other.T;
		this->P				= Other.P;
		this->N				= Other.N;
		this->Wo			= Other.Wo;
		this->Le			= Other.Le;
		this->UV			= Other.UV;
		this->ObjectID		= Other.ObjectID;
		this->LightID		= Other.LightID;

		return *this;
	}

	SampleType	Type;
	bool		Valid;
	float		T;
	Vec3f		P;
	Vec3f		N;
	Vec3f		Wo;
	ColorXYZf	Le;
	Vec2f		UV;
	int			ObjectID;
	int			LightID;
};

}
