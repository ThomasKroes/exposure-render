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

#include "geometry.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL Camera
{
public:
	HOST Camera()
	{
	}

	HOST Camera(const Camera& Other)
	{
		*this = Other;
	}

	HOST ~Camera()
	{
	}
	
	HOST Camera& Camera::operator = (const Camera& Other)
	{
		this->FilmSize		= Other.FilmSize;
		this->Pos			= Other.Pos;
		this->Target		= Other.Target;
		this->Up			= Other.Up;
		this->FocalDistance	= Other.FocalDistance;
		this->ApertureSize	= Other.ApertureSize;
		this->ClipNear		= Other.ClipNear;
		this->ClipFar		= Other.ClipFar;
		this->Exposure		= Other.Exposure;
		this->Gamma			= Other.Gamma;
		this->FOV			= Other.FOV;
		
		this->Update();

		return *this;
	}

	HOST void Update()
	{
		this->InvExposure	= this->Exposure == 0.0f ? 0.0f : 1.0f / this->Exposure;
		this->InvGamma		= this->Gamma == 0.0f ? 0.0f : 1.0f / this->Gamma;
		
		this->N = Normalize(this->Target - this->Pos);
		this->U = Normalize(Cross(this->N, this->Up));
		this->V = Normalize(Cross(this->N, this->U));

		if (this->FocalDistance == -1.0f)
			this->FocalDistance = (this->Target - this->Pos).Length();

		float Scale = 0.0f;

		Scale = tanf((0.5f * this->FOV / RAD_F));

		const float AspectRatio = (float)this->FilmSize[1] / (float)this->FilmSize[0];

		if (AspectRatio > 1.0f)
		{
			this->Screen[0][0] = -Scale;
			this->Screen[0][1] = Scale;
			this->Screen[1][0] = -Scale * AspectRatio;
			this->Screen[1][1] = Scale * AspectRatio;
		}
		else
		{
			this->Screen[0][0] = -Scale / AspectRatio;
			this->Screen[0][1] = Scale / AspectRatio;
			this->Screen[1][0] = -Scale;
			this->Screen[1][1] = Scale;
		}

		this->InvScreen[0] = (this->Screen[0][1] - this->Screen[0][0]) / (float)this->FilmSize[0];
		this->InvScreen[1] = (this->Screen[1][1] - this->Screen[1][0]) / (float)this->FilmSize[1];
	}

	Vec2i	FilmSize;
	Vec3f	Pos;
	Vec3f	Target;
	Vec3f	Up;
	float	FocalDistance;
	float	ApertureSize;
	float	ClipNear;
	float	ClipFar;
	float	Exposure;
	float	Gamma;
	float	FOV;
	Vec3f	N;
	Vec3f	U;
	Vec3f	V;
	float	Screen[2][2];
	float	InvScreen[2];
	float	InvExposure;
	float	InvGamma;
};

}
