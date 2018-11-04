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

#include "ervolume.h"
#include "boundingbox.h"

namespace ExposureRender
{
class EXPOSURE_RENDER_DLL Volume
{

public:
	HOST Volume() :
		BoundingBox(),
		GradientDeltaX(),
		GradientDeltaY(),
		GradientDeltaZ(),
		Spacing(1.0f),
		InvSpacing(1.0f),
		Size(1.0f),
		InvSize(1.0f),
		MinStep(1.0f),
		Voxels(Enums::Device, "Device Voxels")
	{
		DebugLog(__FUNCTION__);
	}

	HOST virtual ~Volume(void)
	{
		DebugLog(__FUNCTION__);
	}

	HOST Volume(const Volume& Other) :
		BoundingBox(),
		GradientDeltaX(),
		GradientDeltaY(),
		GradientDeltaZ(),
		Spacing(1.0f),
		InvSpacing(1.0f),
		Size(1.0f),
		InvSize(1.0f),
		MinStep(1.0f),
		Voxels(Enums::Device, "Device Voxels")
	{
		DebugLog(__FUNCTION__);
		*this = Other;
	}
		
	HOST Volume(const ErVolume& Other) :
		BoundingBox(),
		GradientDeltaX(),
		GradientDeltaY(),
		GradientDeltaZ(),
		Spacing(1.0f),
		InvSpacing(1.0f),
		Size(1.0f),
		InvSize(1.0f),
		MinStep(1.0f),
		Voxels(Enums::Device, "Device Voxels")
	{
		DebugLog(__FUNCTION__);
		*this = Other;
	}

	HOST Volume& Volume::operator = (const Volume& Other)
	{
		DebugLog(__FUNCTION__);

		this->BoundingBox		= Other.BoundingBox;
		this->GradientDeltaX 	= Other.GradientDeltaX;
		this->GradientDeltaY 	= Other.GradientDeltaY;
		this->GradientDeltaZ 	= Other.GradientDeltaZ;
		this->Spacing			= Other.Spacing;
		this->InvSpacing		= Other.InvSpacing;
		this->Size				= Other.Size;
		this->InvSize			= Other.InvSize;
		this->MinStep			= Other.MinStep;
		this->Voxels			= Other.Voxels;

		return *this;
	}

	HOST Volume& Volume::operator = (const ErVolume& Other)
	{
		DebugLog(__FUNCTION__);

		this->Voxels = Other.Voxels;

		float Scale = 0.0f;

		if (Other.NormalizeSize)
		{
			const Vec3f PhysicalSize = Vec3f((float)this->Voxels.Resolution[0], (float)this->Voxels.Resolution[1], (float)this->Voxels.Resolution[2]) * Other.Spacing;
			Scale = 1.0f / max(PhysicalSize[0], max(PhysicalSize[1], PhysicalSize[2]));
		}

		this->Spacing		= Scale * Other.Spacing;
		this->InvSpacing	= 1.0f / this->Spacing;
		this->Size			= Vec3f((float)this->Voxels.Resolution[0] * this->Spacing[0], (float)this->Voxels.Resolution[1] *this->Spacing[1], (float)this->Voxels.Resolution[2] * this->Spacing[2]);
		this->InvSize		= 1.0f / this->Size;

		this->BoundingBox.SetMinP(-0.5 * Size);
		this->BoundingBox.SetMaxP(0.5f * Size);

		this->MinStep = min(this->Spacing[0], min(this->Spacing[1], this->Spacing[2]));

		this->GradientDeltaX = Vec3f(this->MinStep, 0.0f, 0.0f);
		this->GradientDeltaY = Vec3f(0.0f, this->MinStep, 0.0f);
		this->GradientDeltaZ = Vec3f(0.0f, 0.0f, this->MinStep);

		return *this;
	}

	HOST_DEVICE unsigned short operator()(const Vec3f& XYZ = Vec3f(0.0f)) const
	{
		const Vec3f Offset = XYZ - this->BoundingBox.MinP;
		
		const Vec3f LocalXYZ = Offset * this->InvSize * Vec3f(this->Voxels.Resolution[0], this->Voxels.Resolution[1], this->Voxels.Resolution[2]);

		return this->Voxels(LocalXYZ);
	}

	BoundingBox					BoundingBox;
	Vec3f						GradientDeltaX;
	Vec3f						GradientDeltaY;
	Vec3f						GradientDeltaZ;
	Vec3f						Spacing;
	Vec3f						InvSpacing;
	Vec3f						Size;
	Vec3f						InvSize;
	float						MinStep;
	Buffer3D<unsigned short>	Voxels;
};

}
