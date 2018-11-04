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

#include "defines.h"
#include "enums.h"
#include "exception.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL ErBindable
{
public:
	HOST ErBindable()
	{
		this->ID		= -1;
		this->Enabled	= true;
		this->Dirty		= false;
	}

	HOST virtual ~ErBindable()
	{
	}

	HOST ErBindable(const ErBindable& Other)
	{
		*this = Other;
	}

	HOST ErBindable& operator = (const ErBindable& Other)
	{
		this->ID		= Other.ID;
		this->Enabled	= Other.Enabled;
		this->Dirty		= Other.Dirty;

		return *this;
	}

	HOST void BindHost();
	HOST void UnbindHost();

	/*
	GET_SET(ID, int)
	GET_SET(Enabled, bool)
	GET_SET(Dirty, bool)
	*/

	mutable int		ID;
	bool			Enabled;
	bool			Dirty;
};

}
