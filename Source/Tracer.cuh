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
	//Volume<unsigned short>		Volume;
	ErScalarTransferFunction1D		Opacity1D;
	ErColorTransferFunction1D		Diffuse1D;
	ErColorTransferFunction1D		Specular1D;
	ErScalarTransferFunction1D		Glossiness1D;
	ErColorTransferFunction1D		Emission1D;
	/*
	ErLights						Lights;
	ErObjects						Objects;
	ErClippingObjects				ClippingObjects;
	ErTextures						Textures;
	*/
	ErCamera						Camera;
	ErRenderSettings				RenderSettings;
	GaussianFilter					FrameEstimateFilter;
	BilateralFilter					PostProcessingFilter;
	FrameBuffer						FrameBuffer;
	int								NoIterations;

	/*
	std::map<int, ErLight>			LightsMap;
	std::map<int, ErObject>			ObjectsMap;
	std::map<int, ErClippingObject>	ClippingObjectsMap;
	std::map<int, ErTexture>			TexturesMap;

	void CopyLights()
	{
		std::map<int, ExposureRender::ErLight>::iterator It;

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

	void CopyObjects()
	{
		std::map<int, ExposureRender::ErObject>::iterator It;

		Objects.Count = 0;

		for (It = ObjectsMap.begin(); It != ObjectsMap.end(); It++)
		{
			if (It->second.Enabled)
			{
				Objects.List[Objects.Count] = It->second;
				Objects.Count++;
			}
		}
	}

	void CopyClippingObjects()
	{
		std::map<int, ExposureRender::ErClippingObject>::iterator It;

		ClippingObjects.Count = 0;

		for (It = ClippingObjectsMap.begin(); It != ClippingObjectsMap.end(); It++)
		{
			if (It->second.Enabled)
			{
				ClippingObjects.List[ClippingObjects.Count] = It->second;
				ClippingObjects.Count++;
			}
		}
	}

	void CopyTextures()
	{
		std::map<int, ExposureRender::ErTexture>::iterator It;

		Textures.Count = 0;

		for (It = TexturesMap.begin(); It != TexturesMap.end(); It++)
		{
			Textures.List[Textures.Count] = It->second;
			Textures.List[Textures.Count].Image.pData = It->second.Image.pData;
			Textures.Count++;
		}
	}
	*/

	struct Indices
	{
		int										IDs[64];
		int										NoIDs;
		std::map<int, int>						Map;
		static std::map<int, int>::iterator		IndexIt;

		void BindID(int ID)
		{
			this->Map[ID] = ID;
			this->Update();
		}

		void UnbindID(int ID)
		{
			this->IndexIt = this->Map.find(ID);

			const bool Exists = this->IndexIt != this->Map.end();

			if (!Exists)
				return;

			this->Map.erase(IndexIt);
			this->Update();
		}

		void Update()
		{
			this->NoIDs = 0;

			for (this->IndexIt = this->Map.begin(); this->IndexIt != this->Map.end(); this->IndexIt++)
			{
				this->IDs[this->NoIDs] = this-> IndexIt->second;
				this->NoIDs++;
			}
		}
	};

	Indices		VolumeIDs;
	Indices		LightIDs;
	Indices		ObjectIDs;
	Indices		ClippingObjectIDs;

	void BindVolume(int VolumeID)						{	this->VolumeIDs.BindID(VolumeID);						}
	void UnbindVolume(int VolumeID)						{	this->VolumeIDs.UnbindID(VolumeID);						}
	void BindLight(int LightID)							{	this->LightIDs.BindID(LightID);							}
	void UnbindLight(int LightID)						{	this->LightIDs.UnbindID(LightID);						}
	void BindObject(int ObjectID)						{	this->ObjectIDs.BindID(ObjectID);						}
	void UnbindObject(int ObjectID)						{	this->ObjectIDs.UnbindID(ObjectID);						}
	void BindClippingObject(int ClippingObjectID)		{	this->ClippingObjectIDs.BindID(ClippingObjectID);		}
	void UnbindClippingObject(int ClippingObjectID)		{	this->ClippingObjectIDs.UnbindID(ClippingObjectID);		}
};

}
