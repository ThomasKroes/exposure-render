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

#include "erbitmap.h"
#include "buffer.h"

namespace ExposureRender
{

class Bitmap
{
public:
	HOST Bitmap() :
		Pixels(Enums::Device, "Device Pixels")
	{
		DebugLog(__FUNCTION__);
	}

	HOST virtual ~Bitmap(void)
	{
		DebugLog(__FUNCTION__);
	}

	HOST Bitmap(const Bitmap& Other) :
		Pixels(Enums::Device, "Device Pixels")
	{
		DebugLog(__FUNCTION__);
		*this = Other;
	}
		
	HOST Bitmap(const ErBitmap& Other) :
		Pixels(Enums::Device, "Device Pixels")
	{
		DebugLog(__FUNCTION__);
		*this = Other;
	}

	HOST Bitmap& operator = (const Bitmap& Other)
	{
		DebugLog(__FUNCTION__);

		this->Pixels = Other.Pixels;

		return *this;
	}

	HOST Bitmap& operator = (const ErBitmap& Other)
	{
		DebugLog(__FUNCTION__);

		this->Pixels = Other.Pixels;

		return *this;
	}

	Buffer2D<ColorRGBAuc> Pixels;
};

}
