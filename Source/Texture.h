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
#include "Image.h"

namespace ExposureRender
{

struct EXPOSURE_RENDER_DLL Texture
{
	Enums::TextureType	Type;
	float				OutputLevel;
	Image				Image;
	Procedural			Procedural;
	Vec2f				Offset;
	Vec2f				Repeat;
	bool				Flip[2];

	Texture()
	{
	}

	~Texture()
	{
	}

	Texture& operator = (const Texture& Other)
	{
		this->Type			= Other.Type;
		this->OutputLevel	= Other.OutputLevel;
		this->Image			= Other.Image;
		this->Procedural	= Other.Procedural;
		this->Offset		= Other.Offset;
		this->Repeat		= Other.Repeat;
		this->Flip[0]		= Other.Flip[0];
		this->Flip[1]		= Other.Flip[1];
		
		/*
		this->Type			= Other.Type;
		this->OutputLevel	= Other.OutputLevel;
		this->Image			= Other.Image;
		this->Procedural	= Other.Procedural;
		this->Offset[0]		= Other.Offset[0];
		this->Offset[1]		= Other.Offset[1];
		this->Repeat[0]		= Other.Repeat[0];
		this->Repeat[1]		= Other.Repeat[1];
		this->Flip[0]		= Other.Flip[0];
		this->Flip[1]		= Other.Flip[1];

		if (this->Image.Dirty)
		{
			if (this->Image.pData)
				CUDA::Free(this->Image.pData);

			if (this->Image.pData)
			{
				const int NoPixels = this->Image.Size[0] * this->Image.Size[1];
			
				CUDA::Allocate(this->Image.pData, NoPixels);
				CUDA::MemCopyHostToDevice(Other.Image.pData, this->Image.pData, NoPixels);
			}
		} 

		return *this;
		*/
		
		return *this;
	}
};

}
