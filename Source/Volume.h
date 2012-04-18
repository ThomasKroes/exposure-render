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

#include "Vector.h"

#if defined (__CUDA_ARCH__)
	#include "CudaUtilities.h"
#endif

namespace ExposureRender
{

struct Volume
{
	HOST Volume()
	{
	}

	HOST ~Volume()
	{
	}
	
	HOST Volume(const Volume& Other)
	{
		*this = Other;
	}

	HOST Volume& Volume::operator = (const Volume& Other)
	{
		this->Resolution				= Other.Resolution;
		this->InvResolution				= Other.InvResolution;
		this->MinAABB					= Other.MinAABB;
		this->MaxAABB					= Other.MaxAABB;
		this->Size						= Other.Size;
		this->InvSize					= Other.InvSize;
		this->NormalizeSize				= Other.NormalizeSize;
		this->Spacing					= Other.Spacing;
		this->InvSpacing				= Other.InvSpacing;
		this->GradientDeltaX			= Other.GradientDeltaX;
		this->GradientDeltaY			= Other.GradientDeltaY;
		this->GradientDeltaZ			= Other.GradientDeltaZ;
		this->GradientMagnitudeRange	= Other.GradientMagnitudeRange;
		this->pVoxels					= Other.pVoxels;

		return *this;
	}

	HOST void Free()
	{
#ifdef __CUDA_ARCH__
		if (this->pVoxels != NULL)
			CUDA::Free(this->pVoxels);
#endif
	}

	HOST_DEVICE unsigned short Get(const Vec3i& XYZ) const
	{
		if (!this->pVoxels)
			return unsigned short();
		
		Vec3i ClampedXYZ = XYZ;
		ClampedXYZ.Clamp(Vec3i(0, 0, 0), Vec3i(this->Resolution[0] - 1, this->Resolution[1] - 1, this->Resolution[2] - 1));
		
		return this->pVoxels[ClampedXYZ[2] * (int)this->Resolution[0] * (int)this->Resolution[1] + ClampedXYZ[1] * (int)this->Resolution[0] + ClampedXYZ[0]];
	}

	HOST_DEVICE unsigned short Get(const Vec3f& XYZ) const
	{
		Vec3f LocalXYZ = Vec3f((float)this->Resolution[0], (float)this->Resolution[1], (float)this->Resolution[2]) * ((XYZ - this->MinAABB) * this->InvSize);

		return this->Get(Vec3i((int)LocalXYZ[0], (int)LocalXYZ[1], (int)LocalXYZ[2]));
	}
	
	HOST static Volume FromHost(const Volume& Other)
	{
		Volume Result = Other;
		
		float Scale = 1.0f;

		if (Result.NormalizeSize)
		{
			const Vec3f PhysicalSize = Vec3f((float)Result.Resolution[0], (float)Result.Resolution[1], (float)Result.Resolution[2]) * Result.Spacing;
			
			const float Max = max(PhysicalSize[0], max(PhysicalSize[1], PhysicalSize[2]));
			Scale = 1.0f / Max;
		}

		Result.InvResolution[0]	= 1.0f / Result.Resolution[0];
		Result.InvResolution[1]	= 1.0f / Result.Resolution[1];
		Result.InvResolution[2]	= 1.0f / Result.Resolution[2];
		Result.Spacing			= Scale * Result.Spacing;
		Result.InvSpacing[0]	= 1.0f / Result.Spacing[0];
		Result.InvSpacing[1]	= 1.0f / Result.Spacing[1];
		Result.InvSpacing[2]	= 1.0f / Result.Spacing[2];
		Result.Size				= Vec3f((float)Result.Resolution[0], (float)Result.Resolution[1], (float)Result.Resolution[2]) * Result.Spacing;
		Result.InvSize[0]		= 1.0f / Result.Size[0];
		Result.InvSize[1]		= 1.0f / Result.Size[1];
		Result.InvSize[2]		= 1.0f / Result.Size[2];
		Result.MinAABB			= -0.5f * Result.Size;
		Result.MaxAABB			= 0.5f * Result.Size;

		const float MinVoxelSize = min(Result.Spacing[0], min(Result.Spacing[1], Result.Spacing[2]));

		Result.GradientDeltaX[0]	= MinVoxelSize;
		Result.GradientDeltaX[1]	= 0.0f;
		Result.GradientDeltaX[2]	= 0.0f;
		Result.GradientDeltaY[0]	= 0.0f;
		Result.GradientDeltaY[1]	= MinVoxelSize;
		Result.GradientDeltaY[2]	= 0.0f;
		Result.GradientDeltaZ[0]	= 0.0f;
		Result.GradientDeltaZ[1]	= 0.0f;
		Result.GradientDeltaZ[2]	= MinVoxelSize;

#ifdef __CUDA_ARCH__
		// Result.Free();
		
		const int NoVoxels = (int)Result.Resolution[0] * (int)Result.Resolution[1] * (int)Result.Resolution[2];

		if (NoVoxels <= 0)
			return Result;
		
		unsigned short* pDeviceVoxels = NULL;
		CUDA::Allocate(pDeviceVoxels, NoVoxels);

		CUDA::MemCopyHostToDevice(Result.pVoxels, pDeviceVoxels, NoVoxels);
		Result.pVoxels = pDeviceVoxels;
#endif

		return Result;
	}

	Vec3i				Resolution;			// FIXME
	Vec3f				InvResolution;
	Vec3f				MinAABB;
	Vec3f				MaxAABB;
	Vec3f				Size;
	Vec3f				InvSize;
	bool				NormalizeSize;
	Vec3f				Spacing;
	Vec3f				InvSpacing;
	Vec3f				GradientDeltaX;
	Vec3f				GradientDeltaY;
	Vec3f				GradientDeltaZ;
	Vec2f				GradientMagnitudeRange;
	unsigned short*		pVoxels;
};

}
