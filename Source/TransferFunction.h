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

#include "Defines.h"
#include "Enums.h"

namespace ExposureRender
{

struct EXPOSURE_RENDER_DLL PiecewiseLinearFunction
{
	float	NodeRange[2];
	float	Position[MAX_NO_TF_NODES];
	float	Data[MAX_NO_TF_NODES];
	int		Count;

	PiecewiseLinearFunction()
	{
		this->NodeRange[0]	= 0.0f;
		this->NodeRange[1]	= 0.0f;

		for (int i = 0; i < MAX_NO_TF_NODES; i++)
		{
			this->Position[i]	= 0.0f;
			this->Data[i]		= 0.0f;
		}

		this->Count = 0;
	}

	PiecewiseLinearFunction& operator = (const PiecewiseLinearFunction& Other)
	{
		this->NodeRange[0] = Other.NodeRange[0];
		this->NodeRange[1] = Other.NodeRange[1];

		for (int i = 0; i < MAX_NO_TF_NODES; i++)
		{
			this->Position[i]	= Other.Position[i];
			this->Data[i]		= Other.Data[i];
		}	
		
		this->Count = Other.Count;

		return *this;
	}
};

template<int Size>
struct EXPOSURE_RENDER_DLL TransferFunction1D
{
	PiecewiseLinearFunction PLF[Size];
	
	TransferFunction1D& operator = (const TransferFunction1D& Other)
	{	
		for (int i = 0; i < Size; i++)
			this->PLF[i] = Other.PLF[i];
		
		return *this;
	}
};

typedef TransferFunction1D<1> ScalarTransferFunction1D;
typedef TransferFunction1D<3> ColorTransferFunction1D;

}
