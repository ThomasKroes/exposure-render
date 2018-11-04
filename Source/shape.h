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

class EXPOSURE_RENDER_DLL Shape
{
public:
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

	Matrix44			TM;
	Matrix44			InvTM;
	bool				OneSided;
	Enums::ShapeType	Type;
	Vec3f				Size;
	float				Area;
	float				InnerRadius;
	float				OuterRadius;
};

}
