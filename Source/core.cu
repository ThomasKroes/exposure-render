/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define __CUDA_ARCH__ 200


#include "tracer.h"
#include "volume.h"
#include "light.h"
#include "object.h"
#include "clippingobject.h"
#include "texture.h"
#include "bitmap.h"

DEVICE ExposureRender::Tracer*			gpTracer			= NULL;
DEVICE ExposureRender::Volume* 			gpVolumes			= NULL;
DEVICE ExposureRender::Light*			gpLights			= NULL;
DEVICE ExposureRender::Object*			gpObjects			= NULL;
DEVICE ExposureRender::ClippingObject*	gpClippingObjects	= NULL;
DEVICE ExposureRender::Texture*			gpTextures			= NULL;
DEVICE ExposureRender::Bitmap*			gpBitmaps			= NULL;

#include "list.cuh"

ExposureRender::Cuda::List<ExposureRender::Tracer, ExposureRender::ErTracer>					gTracers("gpTracer");
ExposureRender::Cuda::List<ExposureRender::Volume, ExposureRender::ErVolume>					gVolumes("gpVolumes");
ExposureRender::Cuda::List<ExposureRender::Light, ExposureRender::ErLight>						gLights("gpLights");
ExposureRender::Cuda::List<ExposureRender::Object, ExposureRender::ErObject>					gObjects("gpObjects");
ExposureRender::Cuda::List<ExposureRender::ClippingObject, ExposureRender::ErClippingObject>	gClippingObjects("gpClippingObjects");
ExposureRender::Cuda::List<ExposureRender::Texture, ExposureRender::ErTexture>					gTextures("gpTextures");
ExposureRender::Cuda::List<ExposureRender::Bitmap, ExposureRender::ErBitmap>					gBitmaps("gpBitmaps");

#include "singlescattering.cuh"
#include "filterframeestimate.cuh"
#include "estimate.cuh"
#include "tonemap.cuh"

namespace ExposureRender
{

EXPOSURE_RENDER_DLL void BindTracer(const ErTracer& Tracer, const bool& Bind /*= true*/)
{
	DebugLog("%s, Bind = %s", __FUNCTION__, Bind ? "true" : "false");
	
	if (Bind)
		gTracers.Bind(Tracer);
	else
		gTracers.Unbind(Tracer);
}

EXPOSURE_RENDER_DLL void BindVolume(const ErVolume& Volume, const bool& Bind /*= true*/)
{
	DebugLog("%s, Bind = %s", __FUNCTION__, Bind ? "true" : "false");

	if (Bind)
		gVolumes.Bind(Volume);
	else
		gVolumes.Unbind(Volume);
}

EXPOSURE_RENDER_DLL void BindLight(const ErLight& Light, const bool& Bind /*= true*/)
{
	DebugLog("%s, Bind = %s", __FUNCTION__, Bind ? "true" : "false");
	
	if (Bind)
		gLights.Bind(Light);
	else
		gLights.Unbind(Light);
}

EXPOSURE_RENDER_DLL void BindObject(const ErObject& Object, const bool& Bind /*= true*/)
{
	DebugLog("%s, Bind = %s", __FUNCTION__, Bind ? "true" : "false");
	
	if (Bind)
		gObjects.Bind(Object);
	else
		gObjects.Unbind(Object);
}

EXPOSURE_RENDER_DLL void BindClippingObject(const ErClippingObject& ClippingObject, const bool& Bind /*= true*/)
{
	DebugLog("%s, Bind = %s", __FUNCTION__, Bind ? "true" : "false");
	
	if (Bind)
		gClippingObjects.Bind(ClippingObject);
	else
		gClippingObjects.Unbind(ClippingObject);
}

EXPOSURE_RENDER_DLL void BindTexture(const ErTexture& Texture, const bool& Bind /*= true*/)
{
	DebugLog("%s, Bind = %s", __FUNCTION__, Bind ? "true" : "false");
	
	if (Bind)
		gTextures.Bind(Texture);
	else
		gTextures.Unbind(Texture);
}

EXPOSURE_RENDER_DLL void BindBitmap(const ErBitmap& Bitmap, const bool& Bind /*= true*/)
{
	DebugLog("%s, Bind = %s", __FUNCTION__, Bind ? "true" : "false");
	
	if (Bind)
		gBitmaps.Bind(Bitmap);
	else
		gBitmaps.Unbind(Bitmap);
}

EXPOSURE_RENDER_DLL void RenderEstimate(int TracerID)
{
	gTracers.Synchronize(TracerID);

	SingleScattering(gTracers[TracerID]);
	FilterFrameEstimate(gTracers[TracerID]);
	ComputeEstimate(gTracers[TracerID]);
	ToneMap(gTracers[TracerID]);

	gTracers[TracerID].NoIterations++;
}

EXPOSURE_RENDER_DLL void GetEstimate(int TracerID, unsigned char* pData)
{
	FrameBuffer& FB = gTracers[TracerID].FrameBuffer;

	Cuda::MemCopyDeviceToHost(FB.DisplayEstimate.GetData(), (ColorRGBAuc*)pData, FB.DisplayEstimate.GetNoElements());
}

EXPOSURE_RENDER_DLL void GetAutoFocusDistance(int TracerID, int FilmU, int FilmV, float& AutoFocusDistance)
{
//	ComputeAutoFocusDistance(FilmU, FilmV, AutoFocusDistance);
}

EXPOSURE_RENDER_DLL void GetNoIterations(int TracerID, int& NoIterations)
{
//	NoIterations = gTracers[TracerID].NoIterations; 
}

}
