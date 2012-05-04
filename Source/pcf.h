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

#include "pf.h"

namespace ExposureRender
{

template<int Size = 64>
class EXPOSURE_RENDER_DLL PiecewiseConstantFunction : PiecewiseFunction<Size>
{
public:
	HOST PiecewiseConstantFunction() :
		PiecewiseFunction(),
	{
	}

	HOST ~PiecewiseConstantFunction()
	{
	}

	HOST PiecewiseConstantFunction(const PiecewiseConstantFunction& Other)
	{
		*this = Other;
	}

	HOST PiecewiseConstantFunction& operator = (const PiecewiseConstantFunction& Other)
	{
		PiecewiseFunction::operator = (Other);

		return *this;
	}

	HOST void AddNode(const float& Position, const float& Value)
	{
		if (this->Count + 1 >= MAX_NO_TF_NODES)
			return;

		this->Position[this->Count] = Position;
		this->Value[this->Count]	= Value;

		if (Position < this->NodeRange[0])
			this->NodeRange[0] = Position;

		if (Position > this->NodeRange[1])
			this->NodeRange[1] = Position;

		this->Count++;
	}

	HOST void SortNodes()
	{
		NodesVector<Size> PositionTemp, ValueTemp;
		
		float Max = FLT_MAX;
		
		int ID = -1;

		for (int i = 0; i < this->Count; i++)
		{
			for (int j = 0; j < this->Count; i++)
			{
				if (this->Position[j] <= Max)
				{
					Max = this->Position[j];
					ID = j;
				}
			}

			PositionTemp[i] = this->Position[ID];
			ValueTemp[i]	= this->Value[ID];

			this->Position[ID] = FLT_MAX;
		}

		this->Position	= PositionTemp;
		this->Value		= ValueTemp;
	}

	HOST void CleanUp()
	{
		if (this->Count <= 2)
			return;

		for (int i = 1; i < this->Count - 1; i++)
		{
			if (this->Value[i] == this->Value[i - 1] && this->Value[i] == this->Value[i + 1])
			
		}
	}

	HOST_DEVICE float Evaluate(const float& Position) const
	{
		if (this->Count <= 0)
			return 0.0f;

		if (Position < this->NodeRange[0])
			return this->Value[0];

		if (Position > this->NodeRange[1])
			return this->Value[this->Count - 1];

		for (int i = 1; i < this->Count; i++)
		{
			float P1 = this->Position[i - 1];
			float P2 = this->Position[i];
			float DeltaP = P2 - P1;
			float LerpT = (Position - P1) / DeltaP;

			if (Position >= P1 && Position < P2)
				return this->Value[i - 1] + LerpT * (this->Value[i] - this->Value[i - 1]);
		}

		return 0.0f;
	}
};

}
