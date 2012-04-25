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

#include "tracer.h"
#include "volume.h"
#include "light.h"
#include "object.h"
#include "clippingobject.h"
#include "texture.h"
#include "bitmap.h"

namespace ExposureRender
{

EXPOSURE_RENDER_DLL void InitializeTracer(int& ID);
EXPOSURE_RENDER_DLL void DeinitializeTracer(int ID);
EXPOSURE_RENDER_DLL void Resize(int TracerID, int Size[2]);
EXPOSURE_RENDER_DLL void Restart(int TracerID);
EXPOSURE_RENDER_DLL void BindTracer(Tracer Tracer, int& TracerID);
EXPOSURE_RENDER_DLL void UnbindTracer(int TracerID);
EXPOSURE_RENDER_DLL void BindVolume(Volume Volume, int& ID);
EXPOSURE_RENDER_DLL void UnbindVolume(int ID);
EXPOSURE_RENDER_DLL void BindLight(Light Light, int& ID);
EXPOSURE_RENDER_DLL void UnbindLight(int ID);
EXPOSURE_RENDER_DLL void BindObject(Object Object, int& ID);
EXPOSURE_RENDER_DLL void UnbindObject(int ID);
EXPOSURE_RENDER_DLL void BindClippingObject(ClippingObject ClippingObject, int& ID);
EXPOSURE_RENDER_DLL void UnbindClippingObject(int ID);
EXPOSURE_RENDER_DLL void BindTexture(Texture Texture, int& ID);
EXPOSURE_RENDER_DLL void UnbindTexture(int ID);
EXPOSURE_RENDER_DLL void SetVolumeID(int TracerID, int VolumeID);
EXPOSURE_RENDER_DLL void SetLightIDs(int TracerID, Indices LightIDs);
EXPOSURE_RENDER_DLL void SetObjectIDs(int TracerID, Indices ObjectIDs);
EXPOSURE_RENDER_DLL void SetClippingObjectIDs(int TracerID, Indices ClippingObjectIDs);
EXPOSURE_RENDER_DLL void RenderEstimate(int TracerID);
EXPOSURE_RENDER_DLL void GetEstimate(int TracerID, unsigned char* pData);
EXPOSURE_RENDER_DLL void GetAutoFocusDistance(int TracerID, int FilmU, int FilmV, float& AutoFocusDistance);
EXPOSURE_RENDER_DLL void GetNoIterations(int TracerID, int& NoIterations);

}
