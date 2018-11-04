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
#include "shape.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL ErObject : public ErBindable
{
public:
	HOST ErObject() :
		ErBindable(),
		Shape(),
		DiffuseTextureID(-1),
		SpecularTextureID(-1),
		GlossinessTextureID(-1),
		Ior(5.0f)
	{
	}

	HOST virtual ~ErObject()
	{
	}

	HOST ErObject(const ErObject& Other)
	{
		*this = Other;
	}

	HOST ErObject& operator = (const ErObject& Other)
	{
		ErBindable::operator=(Other);

		this->Shape					= Other.Shape;
		this->DiffuseTextureID		= Other.DiffuseTextureID;
		this->SpecularTextureID		= Other.SpecularTextureID;
		this->GlossinessTextureID	= Other.GlossinessTextureID;
		this->Ior					= Other.Ior;

		return *this;
	}

	Shape		Shape;
	int			DiffuseTextureID;
	int			SpecularTextureID;
	int			GlossinessTextureID;
	float		Ior;
};

}
