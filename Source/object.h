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

#include "erobject.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL Object : public ErObject
{
public:
	HOST Object() :
		ErObject()
	{
	}

	HOST Object(const ErObject& Other)
	{
		*this = Other;
	}

	HOST Object& operator = (const ErObject& Other)
	{
		ErObject::operator=(Other);

		return *this;
	}
};

}
