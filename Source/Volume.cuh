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

#include "Defines.cuh"
#include "Vector.cuh"

namespace ExposureRender
{

template <class T>
class Volume
{
public:
	HOST_DEVICE Volume()
	{
	}
	
	HOST_DEVICE T operator[](const int& X, const int& Y, const int& Z) const
	{
		return this->pData[i];
	}

	HOST_DEVICE T operator[](const float& X, const float& Y, const float& Z) const
	{
		return this->pData[i];
	}
	
	Resolution3i		Resolution;
	Resolution3i		InvResolution;
	T*					pData;


	int					Extent[3];
	float				InvExtent[3];
	float				MinAABB[3];
	float				MaxAABB[3];
	float				InvMinAABB[3];
	float				InvMaxAABB[3];
	float				Size[3];
	float				InvSize[3];
	float				Spacing[3];
	float				InvSpacing[3];
	float				GradientDeltaX[3];
	float				GradientDeltaY[3];
	float				GradientDeltaZ[3];
	Range				IntensityRange;
	Range				GradientMagnitudeRange;
};

}
