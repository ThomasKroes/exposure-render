/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "General.cuh"
#include "GridVolume.cuh"
#include "Shape.cuh"

namespace ExposureRender
{

typedef GridVolume<unsigned short> Volume;

struct SharedResources
{
	Volume								Volumes[MAX_NO_VOLUMES];
	Light								Lights[MAX_NO_LIGHTS];
	Object								Objects[MAX_NO_OBJECTS];
	ClippingObject						ClippingObjects[MAX_NO_CLIPPING_OBJECTS];
	Texture								Textures[MAX_NO_TEXTURES];

	int									NoVolumes;
	int									NoLights;
	int									NoObjects;
	int									NoClippingObjects;
	int									NoTextures;

	std::map<int, Volume>				VolumesMap;
	std::map<int, Light>				LightsMap;
	std::map<int, Object>				ObjectsMap;
	std::map<int, ClippingObject>		ClippingObjectsMap;
	std::map<int, Texture>				TexturesMap;

	int									VolumeCounter;
	int									LightCounter;
	int									ObjectCounter;
	int									ClippingObjectCounter;
	int									TextureCounter;

	static std::map<int, Volume>::iterator			VolumeIt;
	static std::map<int, Light>::iterator			LightIt;
	static std::map<int, Object>::iterator			ObjectIt;
	static std::map<int, ClippingObject>::iterator	ClippingObjectIt;
	static std::map<int, Texture>::iterator			TextureIt;

	SharedResources()
	{
		this->NoVolumes					= 0;
		this->NoLights					= 0;
		this->NoObjects					= 0;
		this->NoClippingObjects			= 0;
		this->NoTextures				= 0;
		this->VolumeCounter				= 0;
		this->LightCounter				= 0;
		this->ObjectCounter				= 0;
		this->ClippingObjectCounter		= 0;
		this->TextureCounter			= 0;
	}

	void BindVolume(int Resolution[3], float Spacing[3], unsigned short* pVoxels, int& VolumeID, bool NormalizeSize)
	{
		VolumeID = VolumeCounter;

		VolumesMap[VolumeCounter] = Volume();
		VolumesMap[VolumeCounter].Set(Vec3f(Resolution[0], Resolution[1], Resolution[2]), Vec3f(Spacing[0], Spacing[1], Spacing[2]), pVoxels, NormalizeSize);
		VolumeCounter++;
		
		NoVolumes = 0;

		for (VolumeIt = VolumesMap.begin(); VolumeIt != VolumesMap.end(); VolumeIt++)
		{
			Volumes[NoVolumes] = VolumeIt->second;
			NoVolumes++;
		}
	}

	void BindLight(Light Light, int& LightID)
	{
		LightID = LightCounter;

		LightIt = LightsMap.find(LightID);

		const bool Exists = LightIt != LightsMap.end();

		LightsMap[LightID] = Light;
		LightCounter++;

		Shape& Shape = LightsMap[LightID].Shape;

		switch (Shape.Type)
		{
			case Enums::Plane:		Shape.Area = PlaneArea(Vec2f(Shape.Size[0], Shape.Size[1]));				break;
			case Enums::Disk:		Shape.Area = DiskArea(Shape.OuterRadius);									break;
			case Enums::Ring:		Shape.Area = RingArea(Shape.OuterRadius, Shape.InnerRadius);				break;
			case Enums::Box:		Shape.Area = BoxArea(Vec3f(Shape.Size[0], Shape.Size[1], Shape.Size[2]));	break;
			case Enums::Sphere:		Shape.Area = SphereArea(Shape.OuterRadius);									break;
			case Enums::Cylinder:	Shape.Area = CylinderArea(Shape.OuterRadius, Shape.Size[2]);				break;
		}

		NoLights = 0;

		for (LightIt = LightsMap.begin(); LightIt != LightsMap.end(); LightIt++)
		{
			if (It->second.Enabled)
			{
				Lights[NoLights] = It->second;
				NoLights++;
			}
		}
	}

	void UnbindLight(Light Light)
	{
		LightIt = LightsMap.find(Light);

		const bool Exists = LightIt != LightsMap.end();

		if (!Exists)
			return;

		LightsMap.erase(LightIt);
	
		NoLights = 0;

		for (LightIt = LightsMap.begin(); LightIt != LightsMap.end(); LightIt++)
		{
			if (LightIt->second.Enabled)
			{
				Lights[NoLights] = LightIt->second;
				NoLights++;
			}
		}
	}
	
	void BindObject(Object Object, int& ObjectID)
	{
		ObjectID = ObjectCounter;

		ObjectIt = ObjectsMap.find(ObjectID);

		ObjectsMap[ObjectID] = Object;
		ObjectCounter++;

		NoObjects = 0;

		for (ObjectIt = ObjectsMap.begin(); ObjectIt != ObjectsMap.end(); ObjectIt++)
		{
			if (ObjectIt->second.Enabled)
			{
				Objects[NoObjects] = ObjectIt->second;
				NoObjects++;
			}
		}
	}

	void UnbindObject(Object Object)
	{
		ObjectIt = ObjectsMap.find(Object);

		const bool Exists = ObjectIt != ObjectsMap.end();

		if (!Exists)
			return;

		ObjectsMap.erase(ObjectIt);

		NoObjects = 0;

		for (ObjectIt = ObjectsMap.begin(); ObjectIt != ObjectsMap.end(); ObjectIt++)
		{
			if (ObjectIt->second.Enabled)
			{
				Objects[NoObjects] = ObjectIt->second;
				NoObjects++;
			}
		}
	}

	void BindClippingObject(ClippingObject ClippingObject, int& ClippingObjectID)
	{
		ClippingObjectID = ClippingObjectCounter;

		ClippingObjectIt = ClippingObjectsMap.find(ClippingObjectID);

		ClippingObjectsMap[ClippingObjectID] = ClippingObject;
		ClippingObjectCounter++;

		NoClippingObjects = 0;

		for (ClippingObjectIt = ClippingObjectsMap.begin(); ClippingObjectIt != ClippingObjectsMap.end(); ClippingObjectIt++)
		{
			if (ClippingObjectIt->second.Enabled)
			{
				ClippingObjects[NoClippingObjects] = ClippingObjectIt->second;
				NoClippingObjects++;
			}
		}
	}

	void UnbindClippingObject(ClippingObject ClippingObject)
	{
		ClippingObjectIt = ClippingObjectsMap.find(ClippingObject);

		const bool Exists = ClippingObjectIt != ClippingObjectsMap.end();

		if (!Exists)
			return;

		ClippingObjectsMap.erase(ClippingObjectIt);

		NoClippingObjects = 0;

		for (ClippingObjectIt = ClippingObjectsMap.begin(); ClippingObjectIt != ClippingObjectsMap.end(); ClippingObjectIt++)
		{
			if (ClippingObjectIt->second.Enabled)
			{
				ClippingObjects[NoClippingObjects] = ClippingObjectIt->second;
				NoClippingObjects++;
			}
		}
	}

	void BindTexture(Texture Texture, int& TextureID)
	{
		TextureID = TextureCounter;

		TextureIt = TexturesMap.find(TextureID);

		const bool Exists = TextureIt != TexturesMap.end();

		TexturesMap[TextureID] = Texture;
		TextureCounter++;

		if (Texture.Image.Dirty)
		{
			if (TexturesMap[TextureID].Image.pData)
				CUDA::Free(TexturesMap[TextureID].Image.pData);

			if (Texture.Image.pData)
			{
				const int NoPixels = TexturesMap[TextureID].Image.Size[0] * TexturesMap[TextureID].Image.Size[1];
			
				CUDA::Allocate(TexturesMap[TextureID].Image.pData, NoPixels);
				CUDA::MemCopyHostToDevice(Texture.Image.pData, TexturesMap[TextureID].Image.pData, NoPixels);
			}
		} 

		NoTextures = 0;

		for (TextureIt = TexturesMap.begin(); TextureIt != TexturesMap.end(); TextureIt++)
		{
			Textures[NoTextures] = TextureIt->second;
			Textures[NoTextures].Image.pData = TextureIt->second.Image.pData;
			NoTextures++;
		}
	}

	void UnbindTexture(Texture Texture)
	{
		TextureIt = TexturesMap.find(Texture);

		const bool Exists = TextureIt != TexturesMap.end();

		if (!Exists)
			return;

		if (TextureIt->second.Image.pData)
			CUDA::Free(TextureIt->second.Image.pData);

		TexturesMap.erase(TextureIt);
		
		NoTextures = 0;

		for (TextureIt = TexturesMap.begin(); TextureIt != TexturesMap.end(); TextureIt++)
		{
			Textures[NoTextures] = TextureIt->second;
			Textures[NoTextures].Image.pData = TextureIt->second.Image.pData;
			NoTextures++;
		}
	}

}

}
