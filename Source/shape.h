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

#include "vector.h"
#include "Matrix.h"

namespace ExposureRender
{

EXPOSURE_RENDER_DLL inline float PlaneArea(const Vec2f& Size)
{
	return Size[0] * Size[1];
}

EXPOSURE_RENDER_DLL inline float DiskArea(const float& Radius)
{
	return PI_F * (Radius * Radius);
}

EXPOSURE_RENDER_DLL inline float RingArea(const float& OuterRadius, const float& InnerRadius)
{
	return DiskArea(OuterRadius) - DiskArea(InnerRadius);
}

EXPOSURE_RENDER_DLL inline float BoxArea(const Vec3f& Size)
{
	return (2.0f * Size[0] * Size[1]) + (2.0f * Size[0] * Size[2]) + (2.0f * Size[1] * Size[2]);
}

EXPOSURE_RENDER_DLL inline float SphereArea(const float& Radius)
{
	return 4.0f * PI_F * (Radius * Radius);
}

EXPOSURE_RENDER_DLL inline float CylinderArea(const float& Radius, const float& Height)
{
	return (2.0f * PI_F * (Radius * Radius)) + (2.0f * PI_F * Radius * Height);
}

struct EXPOSURE_RENDER_DLL Shape
{
	Matrix44			TM;
	Matrix44			InvTM;
	bool				OneSided;
	Enums::ShapeType	Type;
	Vec3f				Size;
	float				Area;
	float				InnerRadius;
	float				OuterRadius;

	HOST Shape()
	{
		this->OneSided		= false;
		this->Type			= Enums::Plane;
		this->Area			= 0.0f;
		this->InnerRadius	= 0.0f;
		this->OuterRadius	= 0.0f;
	}

	HOST ~Shape()
	{
	}

	HOST Shape(const Shape& Other)
	{
		*this = Other;
	}
	
	HOST Shape& operator = (const Shape& Other)
	{
		this->TM			= Other.TM;
		this->InvTM			= Other.InvTM;
		this->OneSided		= Other.OneSided;
		this->Type			= Other.Type;
		this->Size			= Other.Size;
		this->Area			= Other.Area;
		this->InnerRadius	= Other.InnerRadius;
		this->OuterRadius	= Other.OuterRadius;

		return *this;
	}

	HOST void Update()
	{
		switch (this->Type)
		{
			case Enums::Plane:		this->Area = PlaneArea(Vec2f(this->Size[0], this->Size[1]));				break;
			case Enums::Disk:		this->Area = DiskArea(this->OuterRadius);									break;
			case Enums::Ring:		this->Area = RingArea(this->OuterRadius, this->InnerRadius);				break;
			case Enums::Box:		this->Area = BoxArea(this->Size);											break;
			case Enums::Sphere:		this->Area = SphereArea(this->OuterRadius);									break;
			case Enums::Cylinder:	this->Area = CylinderArea(this->OuterRadius, this->Size[2]);				break;
		}
	}
};

}
