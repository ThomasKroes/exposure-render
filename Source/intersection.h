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
