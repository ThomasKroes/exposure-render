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

#include "bindable.h"
#include "color.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL NodesVector
{
public:
	CONSTRUCTORS(NodesVector, float, MAX_NO_TF_NODES)
	ALL_OPERATORS(NodesVector, float, MAX_NO_TF_NODES)
	DATA(float, MAX_NO_TF_NODES)
};

class EXPOSURE_RENDER_DLL PiecewiseLinearFunction
{
public:
	Vec2f			NodeRange;
	NodesVector		Position;
	NodesVector		Data;
	int				Count;

	HOST PiecewiseLinearFunction()
	{
		this->Count = 0;
	}

	HOST ~PiecewiseLinearFunction()
	{
	}

	HOST PiecewiseLinearFunction(const PiecewiseLinearFunction& Other)
	{
		*this = Other;
	}

	PiecewiseLinearFunction& operator = (const PiecewiseLinearFunction& Other)
	{
		this->NodeRange		= Other.NodeRange;
		this->Position		= Other.Position;
		this->Data			= Other.Data;
		this->Count			= Other.Count;

		return *this;
	}
};

class EXPOSURE_RENDER_DLL ScalarTransferFunction1D
{
public:
	PiecewiseLinearFunction PLF;
	
	HOST ScalarTransferFunction1D()
	{
	}

	HOST ~ScalarTransferFunction1D()
	{
	}

	HOST ScalarTransferFunction1D(const ScalarTransferFunction1D& Other)
	{
		*this = Other;
	}

	HOST ScalarTransferFunction1D& operator = (const ScalarTransferFunction1D& Other)
	{	
		this->PLF = Other.PLF;
		
		return *this;
	}
};

class EXPOSURE_RENDER_DLL ColorTransferFunction1D
{
public:
	PiecewiseLinearFunction PLF[3];
	
	HOST ColorTransferFunction1D()
	{
	}

	HOST ~ColorTransferFunction1D()
	{
	}

	HOST ColorTransferFunction1D(const ColorTransferFunction1D& Other)
	{
		*this = Other;
	}

	HOST ColorTransferFunction1D& operator = (const ColorTransferFunction1D& Other)
	{	
		for (int i = 0; i < 3; i++)
			this->PLF[i] = Other.PLF[i];
		
		return *this;
	}
};

}
