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
#include "vector.h"
#include "buffer3d.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL ErVolume : public ErBindable
{
public:
	HOST ErVolume() :
		ErBindable(),
		Voxels(Enums::Host, "Host Voxels"),
		NormalizeSize(false),
		Spacing(1.0f)
	{
	}

	HOST virtual ~ErVolume(void)
	{
	}

	HOST ErVolume(const ErVolume& Other) :
		ErBindable(),
		Voxels(Enums::Host, "Host Voxels"),
		NormalizeSize(false),
		Spacing(1.0f)
	{
		*this = Other;
	}

	HOST ErVolume& ErVolume::operator = (const ErVolume& Other)
	{
		ErBindable::operator=(Other);

		this->Voxels		= Other.Voxels;
		this->NormalizeSize	= Other.NormalizeSize;
		this->Spacing		= Other.Spacing;

		return *this;
	}

	HOST void BindVoxels(const Vec3i& Resolution, const Vec3f& Spacing, unsigned short* Voxels, const bool& NormalizeSize = false)
	{
		this->Voxels.Set(Enums::Host, Resolution, Voxels);

		this->NormalizeSize	= NormalizeSize;
		this->Spacing		= Spacing;
	}

	Buffer3D<unsigned short>	Voxels;
	bool						NormalizeSize;
	Vec3f						Spacing;
};

}
