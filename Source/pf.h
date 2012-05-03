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

#include "vector.h"

namespace ExposureRender
{

template<int Size>
class EXPOSURE_RENDER_DLL NodesVector
{
public:
	CONSTRUCTORS(NodesVector, float, Size)
	ALL_OPERATORS(NodesVector, float, Size)
	DATA(float, Size)
};

template<int Size>
class EXPOSURE_RENDER_DLL PiecewiseFunction
{
public:
	HOST PiecewiseFunction() :
		NodeRange(FLT_MAX, FLT_MIN),
		Position(),
		Value(),
		Count(0)
	{
		this->Count = 0;
	}

	HOST ~PiecewiseFunction()
	{
	}

	HOST PiecewiseFunction(const PiecewiseFunction& Other)
	{
		*this = Other;
	}

	HOST PiecewiseFunction& operator = (const PiecewiseFunction& Other)
	{
		this->NodeRange		= Other.NodeRange;
		this->Position		= Other.Position;
		this->Value			= Other.Value;
		this->Count			= Other.Count;

		return *this;
	}

	Vec2f				NodeRange;
	NodesVector<Size>	Position;
	NodesVector<Size>	Value;
	int					Count;
};

}
