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

#include "vector.h"

namespace ExposureRender
{

template<int Size>
class EXPOSURE_RENDER_DLL NodesVector
{
public:
	CONSTRUCTORS(NodesVector, float, Size)
	ALL_OPERATORS(NodesVector, float, Size)
	DATA(float, Size)
};

template<int Size>
class EXPOSURE_RENDER_DLL PiecewiseFunction
{
public:
	HOST PiecewiseFunction() :
		NodeRange(FLT_MAX, FLT_MIN),
		Position(),
		Value(),
		Count(0)
	{
		this->Count = 0;
	}

	HOST ~PiecewiseFunction()
	{
	}

	HOST PiecewiseFunction(const PiecewiseFunction& Other)
	{
		*this = Other;
	}

	HOST PiecewiseFunction& operator = (const PiecewiseFunction& Other)
	{
		this->NodeRange		= Other.NodeRange;
		this->Position		= Other.Position;
		this->Value			= Other.Value;
		this->Count			= Other.Count;

		return *this;
	}

	Vec2f				NodeRange;
	NodesVector<Size>	Position;
	NodesVector<Size>	Value;
	int					Count;
};

}
