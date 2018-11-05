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

	HOST_DEVICE BoundingBox& operator = (const BoundingBox& Other)
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