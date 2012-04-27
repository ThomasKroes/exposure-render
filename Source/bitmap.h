/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "bindable.h"
#include "color.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL ErBitmap : public Bindable
{
public:
	HOST ErBitmap() :
		Bindable(),
		Size(0, 0),
		HostPixels(NULL),
		HostMemoryOwner(false)
	{
	}

	HOST ~ErBitmap()
	{
	}

	HOST ErBitmap(const ErBitmap& Other)
	{
		*this = Other;
	}

	HOST ErBitmap& operator = (const ErBitmap& Other)
	{
		this->Size				= Other.Size;
		this->HostPixels		= Other.HostPixels;
		this->HostMemoryOwner	= Other.HostMemoryOwner;
		
		return *this;
	}

	HOST void BindPixels(const ColorRGBAuc* Pixels, const Vec2i& Size)
	{
		if (Pixels == NULL)
			throw(Exception(Enums::Warning, "BindPixels() failed: pixels pointer is NULL"));

		this->Size = Size;

		this->UnbindPixels();

		const int NoPixels = this->Size[0] * this->Size[1];

		if (NoPixels <= 0)
			throw(Exception(Enums::Warning, "BindPixels() failed: bad no. pixels!"));

		this->HostPixels = new ColorRGBAuc[NoPixels];

		memcpy(this->HostPixels, Pixels, NoPixels * sizeof(ColorRGBAuc));

		this->Dirty				= true;
		this->HostMemoryOwner	= true;
	}

	HOST void UnbindPixels()
	{
		if (!this->HostMemoryOwner)
			return;

		if (this->HostPixels != NULL)
		{
			delete[] this->HostPixels;
			this->HostPixels = NULL;
		}

		this->Dirty = true;
	}

	Vec2i			Size;
	ColorRGBAuc*	HostPixels;
	bool			HostMemoryOwner;
};

}
