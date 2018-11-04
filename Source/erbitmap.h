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
#include "color.h"
#include "buffer2d.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL ErBitmap : public ErBindable
{
public:
	HOST ErBitmap() :
		ErBindable(),
		Pixels(Enums::Host, "Host Pixels")
	{
	}

	HOST virtual ~ErBitmap()
	{
	}

	HOST ErBitmap(const ErBitmap& Other) :
		ErBindable(),
		Pixels(Enums::Host, "Host Pixels")
	{
		*this = Other;
	}

	HOST ErBitmap& operator = (const ErBitmap& Other)
	{
		this->Pixels = Other.Pixels;
		
		return *this;
	}

	HOST void BindPixels(const Vec2i& Resolution, ColorRGBAuc* Pixels)
	{
		this->Pixels.Set(Enums::Host, Resolution, Pixels);
	}

	Buffer2D<ColorRGBAuc> Pixels;
};

}
