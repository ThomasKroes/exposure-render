/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "General.cuh"

namespace ExposureRender
{


EXPOSURE_RENDER_DLL void BindTexture(Texture* pTexture)
{
	if (!pTexture)
		throw(Exception("Texture", "Invalid texture pointer!"));
	
	std::map<int, ExposureRender::Texture>::iterator It;

	It = gTextureMap.find(pTexture->ID);

	gTextureMap[pTexture->ID] = *pTexture;

	if (pTexture->Image.Dirty)
	{
		if (gTextureMap[pTexture->ID].Image.pData)
		{
			CUDA::Free(gTextureMap[pTexture->ID].Image.pData);
			gTextureMap[pTexture->ID].Image.pData = NULL;
		}

		const int NoPixels = gTextureMap[pTexture->ID].Image.Size[0] * gTextureMap[pTexture->ID].Image.Size[1];
		
		CUDA::Allocate(gTextureMap[pTexture->ID].Image.pData, NoPixels);
		CUDA::MemCopyHostToDevice(pTexture->Image.pData, gTextureMap[pTexture->ID].Image.pData, NoPixels);
	} 

	ExposureRender::Textures Textures;

	for (It = gTextureMap.begin(); It != gTextureMap.end(); It++)
	{
		Textures.TextureList[Textures.NoTextures] = It->second;
		Textures.NoTextures++;
	}

	CUDA::HostToConstantDevice(&Textures, "gTextures");
}

EXPOSURE_RENDER_DLL void UnbindTexture(Texture* pTexture)
{
	if (!pTexture)
		throw(Exception("", "Invalid texture pointer!"));

	std::map<int, ExposureRender::Texture>::iterator It;

	It = gTextureMap.find(pTexture->ID);

	if (It == gTextureMap.end())
		return;

	if (It->second.Image.pData)
		CUDA::Free(It->second.Image.pData);

	gTextureMap.erase(It);

	ExposureRender::Textures Textures;

	for (It = gTextureMap.begin(); It != gTextureMap.end(); It++)
	{
		Textures.TextureList[Textures.NoTextures] = It->second;
		Textures.NoTextures++;
	}

	CUDA::HostToConstantDevice(&Textures, "gTextures"); 
}

EXPOSURE_RENDER_DLL void UnbindAllTextures()
{
	std::map<int, ExposureRender::Texture>::iterator It;

	for (It = gTextureMap.begin(); It != gTextureMap.end(); It++)
	{
		if (It->second.Image.pData)
		CUDA::Free(It->second.Image.pData);
	}

	gTextureMap.clear();

	ExposureRender::Textures Textures;
	CUDA::HostToConstantDevice(&Textures, "gTextures");
}

}