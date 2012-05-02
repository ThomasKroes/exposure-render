/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or witDEVut modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software witDEVut specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT DEVLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT DEVLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) DEVWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
		this->Voxels			= Other.Voxels;

		return *this;
	}

	HOST Volume& Volume::operator = (const ErVolume& Other)
	{
		DebugLog(__FUNCTION__);

		this->Voxels = Other.Voxels;

		float Scale = 1.0f;

		if (Other.NormalizeSize)
		{
			const Vec3f PhysicalSize = Vec3f((float)this->Voxels.Resolution[0], (float)this->Voxels.Resolution[1], (float)this->Voxels.Resolution[2]) * Other.Spacing;
			const float Scale = 1.0f / max(PhysicalSize[0], max(PhysicalSize[1], PhysicalSize[2]));
		}

		this->Spacing		= Scale * Other.Spacing;
		this->InvSpacing	= 1.0f / this->Spacing;

		Vec3f Size((float)this->Voxels.Resolution[0] * this->Spacing[0], (float)this->Voxels.Resolution[1] *this->Spacing[1], (float)this->Voxels.Resolution[2] * this->Spacing[2]);
		
		this->BoundingBox.SetMinP(-0.5f);// * Size);
		this->BoundingBox.SetMaxP(0.5f);// * Size);

		const float MinVoxelSize = min(this->Spacing[0], min(this->Spacing[1], this->Spacing[2]));

		this->GradientDeltaX = Vec3f(MinVoxelSize, 0.0f, 0.0f);
		this->GradientDeltaY = Vec3f(0.0f, MinVoxelSize, 0.0f);
		this->GradientDeltaZ = Vec3f(0.0f, 0.0f, MinVoxelSize);

		return *this;
	}

	HOST_DEVICE unsigned short& operator()(const Vec3f& xyz = Vec3f(0.0f)) const
	{
		Vec3f LocalXYZ = Vec3f((float)this->Voxels.Resolution[0], (float)this->Voxels.Resolution[1], (float)this->Voxels.Resolution[2]) * ((xyz - this->BoundingBox.MinP) * this->BoundingBox.InvSize);

		return this->Voxels(Vec3i((int)LocalXYZ[0], (int)LocalXYZ[1], (int)LocalXYZ[2]));
	}

	BoundingBox					BoundingBox;
	Vec3f						GradientDeltaX;
	Vec3f						GradientDeltaY;
	Vec3f						GradientDeltaZ;
	Vec3f						Spacing;
	Vec3f						InvSpacing;
	Buffer3D<unsigned short>	Voxels;
};

}
