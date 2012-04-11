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

struct EXPOSURE_RENDER_DLL ErPiecewiseLinearFunction
{
	ErRange	NodeRange;
	float	Position[MAX_NO_TF_NODES];
	float	Data[MAX_NO_TF_NODES];
	int		Count;

	ErPiecewiseLinearFunction()
	{
		for (int i = 0; i < MAX_NO_TF_NODES; i++)
		{
			this->Position[i]	= 0.0f;
			this->Data[i]		= 0.0f;
		}

		this->Count = 0;
	}

	ErPiecewiseLinearFunction& operator = (const ErPiecewiseLinearFunction& Other)
	{
		this->NodeRange = Other.NodeRange;

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
struct EXPOSURE_RENDER_DLL ErTransferFunction1D
{
	ErPiecewiseLinearFunction		PLF[Size];
	
	ErTransferFunction1D& operator = (const ErTransferFunction1D& Other)
	{	
		for (int i = 0; i < Size; i++)
			this->PLF[i] = Other.PLF[i];
		
		return *this;
	}
};

typedef ErTransferFunction1D<1>	ErScalarTransferFunction1D;
typedef ErTransferFunction1D<3>	ErColorTransferFunction1D;


struct PiecewiseLinearFunction
{
	HOST PiecewiseLinearFunction()
	{
	}

	HOST PiecewiseLinearFunction(const PiecewiseLinearFunction& Other)
	{
		*this = Other;
	}

	HOST ~PiecewiseLinearFunction()
	{
	}

	HOST PiecewiseLinearFunction& PiecewiseLinearFunction::operator = (const PiecewiseLinearFunction& Other)
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

	HOST PiecewiseLinearFunction& PiecewiseLinearFunction::operator = (const ErPiecewiseLinearFunction& Other)
	{
		this->NodeRange[0] = Other.NodeRange.Min;
		this->NodeRange[1] = Other.NodeRange.Min;

		for (int i = 0; i < MAX_NO_TF_NODES; i++)
		{
			this->Position[i]	= Other.Position[i];
			this->Data[i]		= Other.Data[i];
		}	
		
		this->Count = Other.Count;

		return *this;
	}

	DEVICE_NI float Evaluate(const float& Intensity)
	{
		if (Intensity < this->NodeRange[0])
			return this->Data[0];

		if (Intensity > this->NodeRange[1])
			return this->Data[Count - 1];

		for (int i = 1; i < this->Count; i++)
		{
			float P1 = this->Position[i - 1];
			float P2 = this->Position[i];
			float DeltaP = P2 - P1;
			float LerpT = (Intensity - P1) / DeltaP;

			if (Intensity >= P1 && Intensity < P2)
				return Lerp(LerpT, this->Data[i - 1], this->Data[i]);
		}

		return 0.0f;
	}

	float	NodeRange[2];
	float	Position[MAX_NO_TF_NODES];
	float	Data[MAX_NO_TF_NODES];
	int		Count;
};

template<int Size>
struct TransferFunction1D
{
	PiecewiseLinearFunction PLF[Size];
	
	HOST TransferFunction1D()
	{
	}

	HOST TransferFunction1D(const TransferFunction1D& Other)
	{
		*this = Other;
	}

	HOST ~TransferFunction1D()
	{
	}

	HOST TransferFunction1D<Size>& operator = (const TransferFunction1D<Size>& Other)
	{	
		for (int i = 0; i < Size; i++)
			this->PLF[i] = Other.PLF[i];
		
		return *this;
	}

	HOST TransferFunction1D<Size>& operator = (const ErTransferFunction1D<Size>& Other)
	{	
		for (int i = 0; i < Size; i++)
			this->PLF[i] = Other.PLF[i];
		
		return *this;
	}

	DEVICE_NI float Evaluate(const float& Intensity)
	{
		return PLF[0].Evaluate(Intensity);
	}
};

struct ScalarTransferFunction1D : public TransferFunction1D<1>
{
	HOST ScalarTransferFunction1D& operator = (const ErScalarTransferFunction1D& Other)
	{	
		this->PLF[0] = Other.PLF[0];
		return *this;
	}

	HOST ScalarTransferFunction1D& operator = (const ScalarTransferFunction1D& Other)
	{	
		this->PLF[0] = Other.PLF[0];
		return *this;
	}

	DEVICE_NI float Evaluate(const float& Intensity)
	{
		return PLF[0].Evaluate(Intensity);
	}
};

struct ColorTransferFunction1D : public TransferFunction1D<3>
{
	HOST ColorTransferFunction1D& operator = (const ErColorTransferFunction1D& Other)
	{	
		for (int i = 0; i < 3; i++)
			this->PLF[i] = Other.PLF[i];

		return *this;
	}

	HOST ColorTransferFunction1D& operator = (const ColorTransferFunction1D& Other)
	{	
		for (int i = 0; i < 3; i++)
			this->PLF[i] = Other.PLF[i];

		return *this;
	}

	DEVICE_NI ColorXYZf Evaluate(const float& Intensity)
	{
		return ColorXYZf(PLF[0].Evaluate(Intensity), PLF[1].Evaluate(Intensity), PLF[2].Evaluate(Intensity));
	}
};

}
