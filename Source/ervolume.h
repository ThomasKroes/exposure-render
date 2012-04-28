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
#include "buffer.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL ErVolume : public ErBindable
{
public:
	HOST ErVolume() :
		ErBindable(),
		HostVoxels(Enums::Host),
		NormalizeSize(false)
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

		this->HostVoxels	= Other.HostVoxels;
		this->NormalizeSize	= Other.NormalizeSize;

		return *this;
	}

	HOST void BindVoxels(const Vec3i& Resolution, const Vec3f& Spacing, unsigned short* Voxels, const bool& NormalizeSize = false)
	{
		this->HostVoxels	= Buffer3D<unsigned short>(Enums::Host, Resolution, Spacing, Voxels);
		this->NormalizeSize	= NormalizeSize;
	}

	Buffer3D<unsigned short>	HostVoxels;
	bool						NormalizeSize;
};

}
