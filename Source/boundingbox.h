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

class EXPOSURE_RENDER_DLL BoundingBox
{
public:
	HOST_DEVICE BoundingBox() :
		MinP(FLT_MAX),
		MaxP(FLT_MIN),
		Size(0.0f),
		InvSize(0.0f)
	{
	}

	HOST_DEVICE BoundingBox(const Vec3f& MinP, const Vec3f& MaxP) :
		MinP(MinP),
		MaxP(MaxP),
		Size(MaxP - MinP),
		InvSize(1.0f / Size)
	{
	}

	HOST_DEVICE BoundingBox& BoundingBox::operator = (const BoundingBox& Other)
	{
		this->MinP		= Other.MinP;	
		this->MaxP		= Other.MaxP;
		this->Size		= Other.Size;
		this->InvSize	= Other.InvSize;

		return *this;
	}

	HOST_DEVICE void SetMinP(const Vec3f& MinP)
	{
		this->MinP = MinP;
		this->Update();
	}

	HOST_DEVICE void SetMaxP(const Vec3f& MaxP)
	{
		this->MaxP = MaxP;
		this->Update();
	}

	HOST_DEVICE void Update()
	{
		this->Size		= this->MaxP - this->MinP,
		this->InvSize	= 1.0f / Size;
	}

	Vec3f	MinP;
	Vec3f	MaxP;
	Vec3f	Size;
	Vec3f	InvSize;
};

}