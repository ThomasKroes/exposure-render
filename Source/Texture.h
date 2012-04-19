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

#include "Procedural.h"
#include "Bitmap.h"

namespace ExposureRender
{

struct EXPOSURE_RENDER_DLL Texture
{
	int					ID;
	Enums::TextureType	Type;
	float				OutputLevel;
	Bitmap				Bitmap;
	Procedural			Procedural;
	Vec2f				Offset;
	Vec2f				Repeat;
	bool				Flip[2];

	HOST Texture()
	{
		this->ID			= -1;
		this->Type			= Enums::Procedural;
		this->OutputLevel	= 1.0f;
		this->Flip[0]		= false;
		this->Flip[1]		= false;
	}

	HOST ~Texture()
	{
	}
	
	HOST Texture(const Texture& Other)
	{
		*this = Other;
	}

	HOST Texture& operator = (const Texture& Other)
	{
		this->ID			= Other.ID;
		this->Type			= Other.Type;
		this->OutputLevel	= Other.OutputLevel;
		this->Bitmap		= Other.Bitmap;
		this->Procedural	= Other.Procedural;
		this->Offset		= Other.Offset;
		this->Repeat		= Other.Repeat;
		this->Flip[0]		= Other.Flip[0];
		this->Flip[1]		= Other.Flip[1];
		
		return *this;
	}

	HOST void FromHost(const Texture& Other)
	{
#ifdef __CUDA_ARCH__
		if (this->Bitmap.Dirty)
			CUDA::Free(this->Bitmap.pData);
#endif

		*this = Other;

#ifdef __CUDA_ARCH__
		if (this->Bitmap.Dirty && Other.Bitmap.pData)
		{
			const int NoPixels = this->Bitmap.Size[0] * this->Bitmap.Size[1];
			
			this->Bitmap.pData = NULL;

			CUDA::Allocate(this->Bitmap.pData, NoPixels);
			CUDA::MemCopyHostToDevice(Other.Bitmap.pData, this->Bitmap.pData, NoPixels);
		} 
#endif
	}
};

}
