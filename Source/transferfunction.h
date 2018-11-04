/*
    Exposure Render: An interactive photo-realistic volume rendering framework
    Copyright (C) 2011 Thomas Kroes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "plf.h"
#include "color.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL ScalarNode
{
public:
	HOST ScalarNode(float Position, float Value) :
		Position(Position),
		Value(Value)
	{
	}

	HOST ScalarNode() :
		Position(0.0f),
		Value(0.0f)
	{
	}

	HOST ScalarNode(const ScalarNode& Other)
	{
		*this = Other;
	}

	HOST ScalarNode& operator = (const ScalarNode& Other)
	{
		this->Position	= Other.Position;
		this->Value		= Other.Value;

		return *this;
	}

	float	Position;
	float	Value;
};

class EXPOSURE_RENDER_DLL ColorNode
{
public:
	HOST ColorNode()
	{
	}

	HOST ColorNode(const ColorNode& Other)
	{
		*this = Other;
	}

	HOST ColorNode& operator = (const ColorNode& Other)
	{
		for (int i = 0; i < 3; i++)
			this->ScalarNodes[i] = Other.ScalarNodes[i];

		return *this;
	}

	ScalarNode	ScalarNodes[3];
};

class EXPOSURE_RENDER_DLL ScalarTransferFunction1D
{
public:
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

	HOST void AddNode(const ScalarNode& Node)
	{
		this->PLF.AddNode(Node.Position, Node.Value);
	}

	HOST_DEVICE float Evaluate(const float& Intensity) const
	{
		return this->PLF.Evaluate(Intensity);
	}

	PiecewiseLinearFunction<MAX_NO_TF_NODES> PLF;
};

class EXPOSURE_RENDER_DLL ColorTransferFunction1D
{
public:
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

	HOST void AddNode(const ColorNode& Node)
	{
		for (int i = 0; i < 3; i++)
			this->PLF[i].AddNode(Node.ScalarNodes[i].Position, Node.ScalarNodes[i].Value);
	}

	HOST_DEVICE ColorXYZf Evaluate(const float& Intensity) const
	{
		return ColorXYZf(this->PLF[0].Evaluate(Intensity), this->PLF[1].Evaluate(Intensity), this->PLF[2].Evaluate(Intensity));
	}

	PiecewiseLinearFunction<MAX_NO_TF_NODES> PLF[3];
};

}
