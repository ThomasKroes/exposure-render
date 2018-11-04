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

class EXPOSURE_RENDER_DLL ErClippingObject : public ErBindable
{
public:
	HOST ErClippingObject() :
		ErBindable(),
		Shape(),
		Invert(false)
	{
	}
	
	HOST virtual ~ErClippingObject()
	{
	}
	
	HOST ErClippingObject(const ErClippingObject& Other)
	{
		*this = Other;
	}

	HOST ErClippingObject& operator = (const ErClippingObject& Other)
	{
		ErBindable::operator=(Other);

		this->Shape		= Other.Shape;
		this->Invert	= Other.Invert;

		return *this;
	}

	Shape	Shape;
	bool	Invert;
};

}
