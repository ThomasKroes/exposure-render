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
#include "toneMap.cuh"

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
