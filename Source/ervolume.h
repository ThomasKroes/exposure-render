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

#include "erbindable.h"
#include "vector.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL ErVolume : public ErBindable
{
public:
	HOST ErVolume() :
		ErBindable(),
		Resolution(0, 0, 0),
		NormalizeSize(false),
		Spacing(0.0f, 0.0f, 0.0f),
		HostVoxels(NULL),
		HostMemoryOwner(false)
	{
	}

	HOST ~ErVolume()
	{
	}
	
	HOST ErVolume(const ErVolume& Other)
	{
		*this = Other;
	}

	HOST ErVolume& ErVolume::operator = (const ErVolume& Other)
	{
		ErBindable::operator=(Other);

		this->Resolution				= Other.Resolution;
		this->Spacing					= Other.Spacing;
		this->NormalizeSize				= Other.NormalizeSize;
		this->HostVoxels				= Other.HostVoxels;

		return *this;
	}

	HOST void BindVoxels(const unsigned short* Voxels, const Vec3i& Resolution, const Vec3f& Spacing, const bool& NormalizeSize = false)
	{
		if (Voxels == NULL)
			throw(Exception(Enums::Warning, "BindVoxels() failed: voxels pointer is NULL"));

		this->Resolution		= Resolution;
		this->Spacing			= Spacing;
		this->NormalizeSize		= NormalizeSize;

		this->UnbindVoxels();

		const int NoVoxels = this->Resolution[0] * this->Resolution[1] * this->Resolution[2];

		if (NoVoxels <= 0)
			throw(Exception(Enums::Warning, "BindVoxels() failed: bad no. voxels!"));

		this->HostVoxels = new unsigned short[NoVoxels];

		memcpy(this->HostVoxels, Voxels, NoVoxels * sizeof(unsigned short));

		this->Dirty = true;
	}

	HOST void UnbindVoxels()
	{
		if (!this->HostMemoryOwner)
			return;

		if (this->HostVoxels != NULL)
		{
			delete[] this->HostVoxels;
			this->HostVoxels = NULL;
		}

		this->Dirty = true;
	}

	Vec3i				Resolution;
	bool				NormalizeSize;
	Vec3f				Spacing;
	unsigned short*		HostVoxels;
	bool				HostMemoryOwner;
};

}
