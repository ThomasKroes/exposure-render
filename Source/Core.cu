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

#include "ExposureRender.h"

#include "Tracer.cuh"

DEVICE ExposureRender::Tracer*			gpTracer			= NULL;
DEVICE ExposureRender::Volume* 			gpVolumes			= NULL;
DEVICE ExposureRender::Light*			gpLights			= NULL;
DEVICE ExposureRender::Object*			gpObjects			= NULL;
DEVICE ExposureRender::ClippingObject*	gpClippingObjects	= NULL;
DEVICE ExposureRender::Texture*			gpTextures			= NULL;

ExposureRender::CudaList<ExposureRender::Volume>			gVolumes("gpVolumes");
ExposureRender::CudaList<ExposureRender::Light>				gLights("gpLights");
ExposureRender::CudaList<ExposureRender::Object>			gObjects("gpObjects");
ExposureRender::CudaList<ExposureRender::ClippingObject>	gClippingObjects("gpClippingObjects");
ExposureRender::CudaList<ExposureRender::Texture>			gTextures("gpTextures");

#include "Utilities.cuh"

// Kernels
#include "SingleScattering.cuh"
#include "Estimate.cuh"
#include "ToneMap.cuh"

/*
#include "GaussianFilter.cuh"
#include "BilateralFilter.cuh"
#include "MedianFilter.cuh"

#include "GradientMagnitude.cuh"
#include "AutoFocus.cuh"
*/



#define EDIT_TRACER(id)												\
std::map<int, Tracer>::iterator	It;									\
It = gTracers.find(id);												\
if (It == gTracers.end())											\
	throw(Exception(Enums::Error, "Tracer does not exist!"));		\
Tracer& Tracer = gTracers[id];										

DEVICE ExposureRender::Tracer* pTracer = NULL;

void BindTracer(ExposureRender::Tracer& Tracer)
{
	if (pTracer == NULL)
		ExposureRender::CUDA::Allocate(pTracer);
	
	ExposureRender::CUDA::MemCopyHostToDevice(&Tracer, pTracer);
	ExposureRender::CUDA::MemCopyHostToDeviceSymbol(&pTracer, "gpTracer");
}

namespace ExposureRender
{

std::map<int, Tracer> gTracers;

EXPOSURE_RENDER_DLL void Resize(int TracerID, int Size[2])
{
	EDIT_TRACER(TracerID)

	Tracer.FrameBuffer.Resize(Resolution2i(Size[0], Size[1]));
}

EXPOSURE_RENDER_DLL void Reset(int TracerID)
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
	Reset(TracerID);
}

EXPOSURE_RENDER_DLL void SetLightIDs(int TracerID, Indices LightIDs)
{
	EDIT_TRACER(TracerID)
	Tracer.BindLightIDs(LightIDs, gLights.HashMap);
	Reset(TracerID);
}

EXPOSURE_RENDER_DLL void SetObjectIDs(int TracerID, Indices ObjectIDs)
{
	EDIT_TRACER(TracerID)
	Tracer.BindObjectIDs(ObjectIDs, gObjects.HashMap);
	Reset(TracerID);
}

EXPOSURE_RENDER_DLL void SetClippingObjectIDs(int TracerID, Indices ClippingObjectIDs)
{
	EDIT_TRACER(TracerID)
	Tracer.BindClippingObjectIDs(ClippingObjectIDs, gClippingObjects.HashMap);
	Reset(TracerID);
}

EXPOSURE_RENDER_DLL void BindOpacity1D(int TracerID, ScalarTransferFunction1D Opacity1D)
{
	EDIT_TRACER(TracerID)
	Tracer.Opacity1D = Opacity1D;
}

EXPOSURE_RENDER_DLL void BindDiffuse1D(int TracerID, ColorTransferFunction1D Diffuse1D)
{
	EDIT_TRACER(TracerID)
	Tracer.Diffuse1D = Diffuse1D;
}

EXPOSURE_RENDER_DLL void BindSpecular1D(int TracerID, ColorTransferFunction1D Specular1D)
{
	EDIT_TRACER(TracerID)
	Tracer.Specular1D = Specular1D;
}

EXPOSURE_RENDER_DLL void BindGlossiness1D(int TracerID, ScalarTransferFunction1D Glossiness1D)
{
	EDIT_TRACER(TracerID)
	Tracer.Glossiness1D = Glossiness1D;
}

EXPOSURE_RENDER_DLL void BindEmission1D(int TracerID, ColorTransferFunction1D Emission1D)
{
	EDIT_TRACER(TracerID)
	Tracer.Emission1D = Emission1D;
}

EXPOSURE_RENDER_DLL void BindCamera(int TracerID, Camera Camera)
{
	EDIT_TRACER(TracerID)
	Tracer.Camera = Camera;
	Tracer.Camera.ToDevice();
}

EXPOSURE_RENDER_DLL void BindRenderSettings(int TracerID, RenderSettings RenderSettings)
{
	EDIT_TRACER(TracerID)
	Tracer.RenderSettings = RenderSettings;
}

EXPOSURE_RENDER_DLL void RenderEstimate(int TracerID)
{
	EDIT_TRACER(TracerID)

	BindTracer(Tracer);

	SingleScattering(Tracer.FrameBuffer.Resolution[0], Tracer.FrameBuffer.Resolution[1]);
	ComputeEstimate(Tracer.FrameBuffer.Resolution[0], Tracer.FrameBuffer.Resolution[1]);
	ToneMap(Tracer.FrameBuffer.Resolution[0], Tracer.FrameBuffer.Resolution[1]);

	Tracer.NoIterations++; 
}

EXPOSURE_RENDER_DLL void GetEstimate(int TracerID, unsigned char* pData)
{
	EDIT_TRACER(TracerID)
	CUDA::MemCopyDeviceToHost(Tracer.FrameBuffer.CudaDisplayEstimate.GetPtr(), (ColorRGBAuc*)pData, Tracer.FrameBuffer.CudaDisplayEstimate.GetNoElements());
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
