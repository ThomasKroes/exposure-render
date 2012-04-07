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
#include "General.cuh"
#include "CudaUtilities.cuh"
#include "Volume.cuh"

namespace ExposureRender
{

struct Tracer
{
	~Tracer()
	{
	}

	Volume<unsigned short>			Volume;
	ScalarTransferFunction1D		Opacity1D;
	ColorTransferFunction1D			Diffuse1D;
	ColorTransferFunction1D			Specular1D;
	ScalarTransferFunction1D		Glossiness1D;
	ColorTransferFunction1D			Emission1D;
	
	Lights							Lights;
	Textures						Textures;

	std::map<int, Light>			LightsMap;
	std::map<int, Texture>			TexturesMap;

	void CopyLights()
	{
		std::map<int, ExposureRender::Light>::iterator It;

		Lights.Count = 0;

		for (It = LightsMap.begin(); It != LightsMap.end(); It++)
		{
			if (It->second.Enabled)
			{
				Lights.List[Lights.Count] = It->second;
				Lights.Count++;
			}
		}
	}

	void CopyTextures()
	{
		std::map<int, ExposureRender::Texture>::iterator It;

		Textures.Count = 0;

		for (It = TexturesMap.begin(); It != TexturesMap.end(); It++)
		{
			Textures.List[Textures.Count] = It->second;
			Textures.List[Textures.Count].Image.pData = It->second.Image.pData;
			Textures.Count++;
		}
	}
};

}
