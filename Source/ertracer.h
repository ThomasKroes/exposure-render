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

#include "erbindable.h"
#include "transferfunction.h"
#include "camera.h"
#include "rendersettings.h"

#include <map>

using namespace std;

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL ErTracer : public ErBindable
{
public:
	HOST ErTracer() :
		ErBindable(),
		Opacity1D(),
		Diffuse1D(),
		Specular1D(),
		Glossiness1D(),
		Emission1D(),
		Camera(),
		RenderSettings(),
		NoIterations(0),
		VolumeID(0),
		LightIDs(),
		ObjectIDs(),
		ClippingObjectIDs()
	{
	}

	HOST virtual ~ErTracer()
	{
	}
	
	HOST ErTracer(const ErTracer& Other)
	{
		*this = Other;
	}

	HOST ErTracer& operator = (const ErTracer& Other)
	{
		ErBindable::operator=(Other);

		this->Opacity1D				= Other.Opacity1D;
		this->Diffuse1D				= Other.Diffuse1D;
		this->Specular1D			= Other.Specular1D;
		this->Glossiness1D			= Other.Glossiness1D;
		this->Emission1D			= Other.Emission1D;
		this->Camera				= Other.Camera;
		this->RenderSettings		= Other.RenderSettings;
		this->NoIterations			= Other.NoIterations;
		this->VolumeID				= Other.VolumeID;
		this->LightIDs				= Other.LightIDs;
		this->ObjectIDs				= Other.ObjectIDs;
		this->ClippingObjectIDs		= Other.ClippingObjectIDs;

		return *this;
	}
	
	HOST void BindIDs(Indices SourceIDs, Indices& TargetIDs, map<int, int> HashMap)
	{
		for (int i = 0; i < SourceIDs.Count; i++)
			TargetIDs[i] = HashMap[SourceIDs[i]];

		TargetIDs.Count = SourceIDs.Count;
	}

	HOST void BindLightIDs(Indices LightIDs, map<int, int> HashMap)
	{
		BindIDs(LightIDs, this->LightIDs, HashMap);
	}

	HOST void BindObjectIDs(Indices ObjectIDs, map<int, int> HashMap)
	{
		BindIDs(ObjectIDs, this->ObjectIDs, HashMap);
	}

	HOST void BindClippingObjectIDs(Indices ClippingObjectIDs, map<int, int> HashMap)
	{
		BindIDs(ClippingObjectIDs, this->ClippingObjectIDs, HashMap);
	}

	ScalarTransferFunction1D	Opacity1D;
	ColorTransferFunction1D		Diffuse1D;
	ColorTransferFunction1D		Specular1D;
	ScalarTransferFunction1D	Glossiness1D;
	ColorTransferFunction1D		Emission1D;
	Camera						Camera;
	RenderSettings				RenderSettings;
	int							NoIterations;
	int							VolumeID;
	Indices						LightIDs;
	Indices						ObjectIDs;
	Indices						ClippingObjectIDs;
};

}
