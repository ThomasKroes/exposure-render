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

#include "Shape.h"

namespace ExposureRender
{

struct EXPOSURE_RENDER_DLL Light
{
	int						ID;
	bool					Enabled;
	bool					Visible;
	Shape					Shape;
	int						TextureID;
	float					Multiplier;
	Enums::EmissionUnit		Unit;
	
	HOST Light()
	{
		this->ID			= -1;
		this->Enabled		= true;
		this->Visible		= true;
		this->TextureID		= 0;
		this->Multiplier	= 1.0f;
		this->Unit			= Enums::Power;
	}

	HOST ~Light()
	{
	}

	HOST Light(const Light& Other)
	{
		*this = Other;
	}
	
	HOST Light& operator = (const Light& Other)
	{
		this->ID			= Other.ID;
		this->Enabled		= Other.Enabled;
		this->Visible		= Other.Visible;
		this->Shape			= Other.Shape;
		this->TextureID		= Other.TextureID;
		this->Multiplier	= Other.Multiplier;
		this->Unit			= Other.Unit;

		return *this;
	}

	HOST static Light FromHost(const Light& Other)
	{
		Light Result = Other;
		Result.Shape.Update();
		return Result;
	}
};

}
