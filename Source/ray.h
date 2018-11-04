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

class Ray
{	
public:
	HOST_DEVICE Ray(Vec3f O = Vec3f(), Vec3f D = Vec3f(0.0f, 0.0f, 1.0f), float MinT = 0.0f, float MaxT = 1000000.0f)
	{
		this->O		= O;
		this->D		= D;
		this->MinT	= MinT;
		this->MaxT	= MaxT;
	}

	HOST_DEVICE Ray& operator = (const Ray& Other)
	{
		this->O		= Other.O;
		this->D		= Other.D;
		this->MinT	= Other.MinT;
		this->MaxT	= Other.MaxT;

		return *this;
	}

	HOST_DEVICE Vec3f operator()(float T) const
	{
		return this->O + Normalize(this->D) * T;
	}

	Vec3f 	O;
	Vec3f 	D;
	float	MinT;
	float	MaxT;		
};

}