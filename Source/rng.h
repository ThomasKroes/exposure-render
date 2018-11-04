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

class CRNG
{
public:
	HOST_DEVICE CRNG(unsigned int* pSeed0, unsigned int* pSeed1)
	{
		this->pSeed0 = pSeed0;
		this->pSeed1 = pSeed1;
	}

	HOST_DEVICE float Get1(void)
	{
		*this->pSeed0 = 36969 * ((*pSeed0) & 65535) + ((*pSeed0) >> 16);
		*this->pSeed1 = 18000 * ((*pSeed1) & 65535) + ((*pSeed1) >> 16);

		unsigned int ires = ((*pSeed0) << 16) + (*pSeed1);

		union
		{
			float f;
			unsigned int ui;
		} res;

		res.ui = (ires & 0x007fffff) | 0x40000000;

		return (res.f - 2.f) / 2.f;
	}

	HOST_DEVICE Vec2f Get2(void)
	{
		return Vec2f(Get1(), Get1());
	}

	HOST_DEVICE Vec3f Get3(void)
	{
		return Vec3f(Get1(), Get1(), Get1());
	}

private:
	unsigned int*	pSeed0;
	unsigned int*	pSeed1;
};

}