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

#include "erbindable.h"
#include "procedural.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL ErTexture : public ErBindable
{
public:
	HOST ErTexture() :
		ErBindable(),
		Type(Enums::Procedural),
		OutputLevel(1.0f),
		BitmapID(-1),
		Procedural(),
		Offset(0.0f),
		Repeat(0.0f),
		Flip(0)
	{
	}

	HOST virtual ~ErTexture()
	{
	}
	
	HOST ErTexture(const ErTexture& Other)
	{
		*this = Other;
	}

	HOST ErTexture& operator = (const ErTexture& Other)
	{
		ErBindable::operator=(Other);

		this->Type			= Other.Type;
		this->OutputLevel	= Other.OutputLevel;
		this->BitmapID		= Other.BitmapID;
		this->Procedural	= Other.Procedural;
		this->Offset		= Other.Offset;
		this->Repeat		= Other.Repeat;
		this->Flip			= Other.Flip;
		
		return *this;
	}

	HOST void BindDevice(const ErTexture& HostTexture)
	{
		*this = HostTexture;
	}

	HOST void UnbindDevice()
	{
	}

	Enums::TextureType	Type;
	float				OutputLevel;
	int					BitmapID;
	Procedural			Procedural;
	Vec2f				Offset;
	Vec2f				Repeat;
	Vec2i				Flip;
};

}
