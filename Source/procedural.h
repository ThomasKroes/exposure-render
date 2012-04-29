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

#include "vector.h"
#include "transferfunction.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL Procedural
{
public:
	HOST Procedural()
	{
		this->Type = Enums::Uniform;
	}
	
	HOST ~Procedural()
	{
	}

	HOST Procedural(const Procedural& Other)
	{
		*this = Other;
	}

	HOST Procedural& operator = (const Procedural& Other)
	{
		this->Type			= Other.Type;
		this->UniformColor	= Other.UniformColor;
		this->CheckerColor1	= Other.CheckerColor1;
		this->CheckerColor2	= Other.CheckerColor2;
		this->Gradient		= Other.Gradient;

		return *this;
	}

	Enums::ProceduralType		Type;
	ColorXYZf					UniformColor;
	ColorXYZf					CheckerColor1;
	ColorXYZf					CheckerColor2;
	ColorTransferFunction1D		Gradient;
};

}
