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

#include <cstdio>
#include "defines.h"
#include "enums.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL Exception
{
public:
	HOST Exception(const Enums::ExceptionLevel& Level, const char* pMessage = "")
	{
		this->Level = Level;
		snprintf(this->Message, MAX_CHAR_SIZE, "%s", pMessage);
	}

	HOST ~Exception()
	{
	}

	HOST Exception(const Exception& Other)
	{
		*this = Other;
	}

	HOST Exception& operator = (const Exception& Other)
	{
		this->Level = Other.Level;
		snprintf(this->Message, MAX_CHAR_SIZE, "%s", Other.Message);

		return *this;
	}

	Enums::ExceptionLevel	Level;
	char					Message[MAX_CHAR_SIZE];
};

}
