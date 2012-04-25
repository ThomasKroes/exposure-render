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

#include "exposurerender.h"

#include "tracer.h"

DEVICE ExposureRender::Tracer*			gpTracer			= NULL;
DEVICE ExposureRender::Volume* 			gpVolumes			= NULL;
DEVICE ExposureRender::Light*			gpLights			= NULL;
DEVICE ExposureRender::Object*			gpObjects			= NULL;
DEVICE ExposureRender::ClippingObject*	gpClippingObjects	= NULL;
DEVICE ExposureRender::Texture*			gpTextures			= NULL;
DEVICE ExposureRender::Bitmap*			gpBitmaps			= NULL;

ExposureRender::Cuda::List<ExposureRender::Volume>			gVolumes("gpVolumes");
ExposureRender::Cuda::List<ExposureRender::Light>			gLights("gpLights");
ExposureRender::Cuda::List<ExposureRender::Object>			gObjects("gpObjects");
ExposureRender::Cuda::List<ExposureRender::ClippingObject>	gClippingObjects("gpClippingObjects");
ExposureRender::Cuda::List<ExposureRender::Texture>			gTextures("gpTextures");
ExposureRender::Cuda::List<ExposureRender::Bitmap>			gBitmaps("gpBitmaps");

#include "utilities.h"

#include "singlescattering.cuh"
#include "estimate.cuh"
#include "toneMap.cuh"
#include "gaussianfilter.cuh"

#define EDIT_TRACER(id)												\
std::map<int, Tracer>::iterator	It;									\
It = gTracers.find(id);												\
if (It == gTracers.end())											\
	throw(Exception(Enums::Error, "Tracer does not exist!"));		\
Tracer& Tracer = gTracers[id];										

DEVICE ExposureRender::Tracer* pTracer = NULL;

void BindDeviceTracer(ExposureRender::Tracer& Tracer)
{
	if (pTracer == NULL)
		ExposureRender::Cuda::Allocate(pTracer);
	
	ExposureRender::Cuda::MemCopyHostToDevice(&Tracer, pTracer);
	ExposureRender::Cuda::MemCopyHostToDeviceSymbol(&pTracer, "gpTracer");
}

namespace ExposureRender
{

std::map<int, Tracer> gTracers;

EXPOSURE_RENDER_DLL void Resize(int TracerID, int Size[2])
{
	EDIT_TRACER(TracerID)

	Tracer.FrameBuffer.Resize(Resolution2i(Size[0], Size[1]));
}

EXPOSURE_RENDER_DLL void Restart(int TracerID)
{
	EDIT_TRACER(TracerID)

	Tracer.FrameBuffer.Reset();
	Tracer.NoIterations = 0;
}

EXPOSURE_RENDER_DLL void InitializeTracer(int& ID)
{
	if (ID < 0)
		ID = gTracers.size();

	gTracers[ID] = Tracer();

	gVolumes.Synchronize();
	gLights.Synchronize();
	gObjects.Synchronize();
	gClippingObjects.Synchronize();
}

EXPOSURE_RENDER_DLL void DeinitializeTracer(int ID)
{
	std::map<int, Tracer>::iterator	It;
	
	It = gTracers.find(ID);
	
	if (It != gTracers.end())
		gTracers.erase(ID);
}

EXPOSURE_RENDER_DLL void BindTracer(Tracer T, int& TracerID)
{
	EDIT_TRACER(TracerID)
	Tracer = T;
	Restart(TracerID);
}

EXPOSURE_RENDER_DLL void UnbindTracer(int TracerID)
{
}

EXPOSURE_RENDER_DLL void BindVolume(Volume V, int& ID)
{
	gVolumes.Bind(V, ID);
}

EXPOSURE_RENDER_DLL void UnbindVolume(int ID)
{
	gVolumes.Unbind(ID);
}

EXPOSURE_RENDER_DLL void BindLight(Light L, int& ID)
{
	gLights.Bind(L, ID);
}

EXPOSURE_RENDER_DLL void UnbindLight(int ID)
{
	gLights.Unbind(ID);
}

EXPOSURE_RENDER_DLL void BindObject(Object O, int& ID)
{
	gObjects.Bind(O, ID);
}

EXPOSURE_RENDER_DLL void UnbindObject(int ID)
{
	gObjects.Unbind(ID);
}

EXPOSURE_RENDER_DLL void BindClippingObject(ClippingObject C, int& ID)
{
	gClippingObjects.Bind(C, ID);
}

EXPOSURE_RENDER_DLL void UnbindClippingObject(int ID)
{
	gClippingObjects.Unbind(ID);
}

EXPOSURE_RENDER_DLL void BindTexture(Texture Texture, int& ID)
{
	gTextures.Bind(Texture, ID);
}

EXPOSURE_RENDER_DLL void UnbindTexture(int ID)
{
	gTextures.Unbind(ID);
}

EXPOSURE_RENDER_DLL void SetVolumeID(int TracerID, int VolumeID)
{
	EDIT_TRACER(TracerID)
	Tracer.VolumeID = VolumeID;
}

EXPOSURE_RENDER_DLL void SetLightIDs(int TracerID, Indices LightIDs)
{
	EDIT_TRACER(TracerID)
	Tracer.BindLightIDs(LightIDs, gLights.HashMap);
}

EXPOSURE_RENDER_DLL void SetObjectIDs(int TracerID, Indices ObjectIDs)
{
	EDIT_TRACER(TracerID)
	Tracer.BindObjectIDs(ObjectIDs, gObjects.HashMap);
}

EXPOSURE_RENDER_DLL void SetClippingObjectIDs(int TracerID, Indices ClippingObjectIDs)
{
	EDIT_TRACER(TracerID)
	Tracer.BindClippingObjectIDs(ClippingObjectIDs, gClippingObjects.HashMap);
}

EXPOSURE_RENDER_DLL void RenderEstimate(int TracerID)
{
	EDIT_TRACER(TracerID)

	BindDeviceTracer(Tracer);

	SingleScattering(Tracer.FrameBuffer.Resolution[0], Tracer.FrameBuffer.Resolution[1]);
	ComputeEstimate(Tracer.FrameBuffer.Resolution[0], Tracer.FrameBuffer.Resolution[1]);
	FilterGaussian(Tracer.FrameBuffer.CudaFrameEstimate.GetPtr(), Tracer.FrameBuffer.CudaFrameEstimateTemp.GetPtr(), Tracer.FrameBuffer.Resolution[0], Tracer.FrameBuffer.Resolution[1]);
	ToneMap(Tracer.FrameBuffer.Resolution[0], Tracer.FrameBuffer.Resolution[1]);

	Tracer.NoIterations++;
}

EXPOSURE_RENDER_DLL void GetEstimate(int TracerID, unsigned char* pData)
{
	EDIT_TRACER(TracerID)
	Cuda::MemCopyDeviceToHost(Tracer.FrameBuffer.CudaDisplayEstimate.GetPtr(), (ColorRGBAuc*)pData, Tracer.FrameBuffer.CudaDisplayEstimate.GetNoElements());
}

EXPOSURE_RENDER_DLL void GetAutoFocusDistance(int TracerID, int FilmU, int FilmV, float& AutoFocusDistance)
{
	return;
//	ComputeAutoFocusDistance(FilmU, FilmV, AutoFocusDistance);
}

EXPOSURE_RENDER_DLL void GetNoIterations(int TracerID, int& NoIterations)
{
	EDIT_TRACER(TracerID)
	NoIterations = Tracer.NoIterations; 
}

}
