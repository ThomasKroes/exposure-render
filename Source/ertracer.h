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
