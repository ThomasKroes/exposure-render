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

#include "Core.cuh"

#include "Tracer.cuh"
#include "Volume.cuh"
#include "Light.cuh"
#include "Object.cuh"
#include "ClippingObject.cuh"
#include "Texture.cuh"
#include "Utilities.cuh"

#include <map>

static std::map<int, ExposureRender::Tracer>			gTracers;
static std::map<int, ExposureRender::Volume>			gVolumes;
static std::map<int, ExposureRender::Light>				gLights;
static std::map<int, ExposureRender::Object>			gObjects;
static std::map<int, ExposureRender::ClippingObject>	gClippingObjects;
static std::map<int, ExposureRender::Texture>			gTextures;

static int gTracerCounter			= 0;
static int gVolumeCounter			= 0;
static int gLightCounter			= 0;
static int gObjectCounter			= 0;
static int gClippingObjectCounter	= 0;
static int gTextureCounter			= 0;

#include "GaussianFilter.cuh"
#include "BilateralFilter.cuh"
#include "MedianFilter.cuh"
#include "Estimate.cuh"

#include "SingleScattering.cuh"
#include "ToneMap.cuh"
#include "GradientMagnitude.cuh"
#include "AutoFocus.cuh"

ExposureRender::Tracer* gpCurrentTracer = NULL;

namespace ExposureRender
{

#define EDIT_TRACER(id)											\
																\
std::map<int, Tracer>::iterator TracerIt;						\
																\
TracerIt = gTracers.find(id);									\
																\
const bool TracerExists = TracerIt != gTracers.end();			\
																\
if (!TracerExists)												\
	throw(ErException("CUDA", "Tracer does not exist", ""));	\
																\
Tracer& CurrentTracer = TracerIt->second;




void BindTracer(int TracerID)
{
	EDIT_TRACER(TracerID)

	if (gpCurrentTracer == NULL)
		cudaMalloc(&gpCurrentTracer, sizeof(Tracer));

	cudaMemcpy(gpCurrentTracer, &CurrentTracer, sizeof(Tracer), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(gpTracer, &gpCurrentTracer, sizeof(gpCurrentTracer));
}

void UnbindTracer(int TracerID)
{
	EDIT_TRACER(TracerID)

	gTracers.erase(TracerIt);
	
	if (gTracers.empty())
		CUDA::Free(gpCurrentTracer);
}

EXPOSURE_RENDER_DLL void Resize(int TracerID, int Size[2])
{
	EDIT_TRACER(TracerID)

	CurrentTracer.FrameBuffer.Resize(Resolution2i(Size));

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void Reset(int TracerID)
{
	EDIT_TRACER(TracerID)

	CurrentTracer.FrameBuffer.Reset();
	CurrentTracer.NoIterations = 0;

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void InitializeTracer(int& TracerID)
{
	TracerID = gTracerCounter;
	gTracers[gTracerCounter] = Tracer();
	gTracerCounter++;

	EDIT_TRACER(TracerID)

	CurrentTracer.FrameBuffer.Free();

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void DeinitializeTracer(int TracerID)
{
	
}

EXPOSURE_RENDER_DLL void BindVolume(ErVolume Volume, int& ID)
{
	ID = gVolumeCounter;
	gVolumes[gVolumeCounter] = Volume;
	gVolumeCounter++;
}

EXPOSURE_RENDER_DLL void UnbindVolume(int ID)
{
}

EXPOSURE_RENDER_DLL void BindLight(ErLight Light, int& ID)
{
	ID = gLightCounter;
	gLights[gLightCounter] = Light;
	gLightCounter++;
}

EXPOSURE_RENDER_DLL void UnbindLight(int ID)
{
	std::map<int, ExposureRender::Light>::iterator It;

	It = gLights.find(ID);

	if (It != gLights.end())
		gLights.erase(It);
}

EXPOSURE_RENDER_DLL void BindObject(ErObject Object, int& ID)
{
	ID = gObjectCounter;
	gObjects[gObjectCounter] = Object;
	gObjectCounter++;
}

EXPOSURE_RENDER_DLL void UnbindObject(int ID)
{
	std::map<int, ExposureRender::Object>::iterator It;

	It = gObjects.find(ID);

	if (It != gObjects.end())
		gObjects.erase(It);
}

EXPOSURE_RENDER_DLL void BindClippingObject(ErClippingObject ClippingObject, int& ID)
{
	ID = gClippingObjectCounter;
	gClippingObjects[gClippingObjectCounter] = ClippingObject;
	gClippingObjectCounter++;
}

EXPOSURE_RENDER_DLL void UnbindClippingObject(int ID)
{
	std::map<int, ExposureRender::ClippingObject>::iterator It;

	It = gClippingObjects.find(ID);

	if (It != gClippingObjects.end())
		gClippingObjects.erase(It);
}

EXPOSURE_RENDER_DLL void BindTexture(ErTexture Texture, int& ID)
{
	ID = gTextureCounter;
	gTextures[gTextureCounter] = Texture;
	gTextureCounter++;
}

EXPOSURE_RENDER_DLL void UnbindTexture(int ID)
{
	std::map<int, ExposureRender::Texture>::iterator It;

	It = gTextures.find(ID);

	if (It != gTextures.end())
		gTextures.erase(It);
}

EXPOSURE_RENDER_DLL void SetTracerVolumeIDs(int ID[MAX_NO_VOLUMES], int Size)
{
}

EXPOSURE_RENDER_DLL void SetTracerLightIDs(int ID[MAX_NO_LIGHTS], int Size)
{
}

EXPOSURE_RENDER_DLL void SetTracerObjectIDs(int ID[MAX_NO_OBJECTS], int Size)
{
}

EXPOSURE_RENDER_DLL void SetTracerClippingObjectIDs(int ID[MAX_NO_CLIPPING_OBJECTS], int Size)
{
}

EXPOSURE_RENDER_DLL void BindOpacity1D(int TracerID, ErScalarTransferFunction1D Opacity1D)
{
	EDIT_TRACER(TracerID)

	CurrentTracer.Opacity1D = Opacity1D;

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void BindDiffuse1D(int TracerID, ErColorTransferFunction1D Diffuse1D)
{
	EDIT_TRACER(TracerID)

	CurrentTracer.Diffuse1D = Diffuse1D;
	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void BindSpecular1D(int TracerID, ErColorTransferFunction1D Specular1D)
{
	EDIT_TRACER(TracerID)

	CurrentTracer.Specular1D = Specular1D;
	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void BindGlossiness1D(int TracerID, ErScalarTransferFunction1D Glossiness1D)
{
	EDIT_TRACER(TracerID)

	CurrentTracer.Glossiness1D = Glossiness1D;
	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void BindEmission1D(int TracerID, ErColorTransferFunction1D Emission1D)
{
	EDIT_TRACER(TracerID)

	CurrentTracer.Emission1D = Emission1D;
	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void BindCamera(int TracerID, ErCamera Camera)
{
	EDIT_TRACER(TracerID)

	CurrentTracer.Camera = Camera;

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void BindRenderSettings(int TracerID, ErRenderSettings RenderSettings)
{
	EDIT_TRACER(TracerID)

	CurrentTracer.RenderSettings = RenderSettings;
	
	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void BindFiltering(int TracerID, ErFiltering Filtering)
{
	EDIT_TRACER(TracerID)

	// Frame estimate filter
	GaussianFilter Gaussian;
	
	Gaussian.KernelRadius = Filtering.FrameEstimateFilter.KernelRadius;

	const int KernelSize = (2 * Gaussian.KernelRadius) + 1;

	for (int i = 0; i < KernelSize; i++)
		Gaussian.KernelD[i] = Gauss2D(Filtering.FrameEstimateFilter.Sigma, Gaussian.KernelRadius - i, 0);

	CurrentTracer.FrameEstimateFilter = Gaussian;
	
	// Post processing filter
	BilateralFilter Bilateral;

	const int SigmaMax = (int)max(Filtering.PostProcessingFilter.SigmaD, Filtering.PostProcessingFilter.SigmaR);
	
	Bilateral.KernelRadius = (int)ceilf(2.0f * (float)SigmaMax);  

	const float TwoSigmaRSquared = 2 * Filtering.PostProcessingFilter.SigmaR * Filtering.PostProcessingFilter.SigmaR;

	const int kernelSize = Bilateral.KernelRadius * 2 + 1;
	const int center = (kernelSize - 1) / 2;

	for (int x = -center; x < -center + kernelSize; x++)
		Bilateral.KernelD[x + center] = Gauss2D(Filtering.PostProcessingFilter.SigmaD, x, 0);

	for (int i = 0; i < 256; i++)
		Bilateral.GaussSimilarity[i] = expf(-((float)i / TwoSigmaRSquared));

	CurrentTracer.PostProcessingFilter = Bilateral;

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void RenderEstimate(int TracerID)
{
	return;

	EDIT_TRACER(TracerID)

	CUDA::ThreadSynchronize();

	SingleScattering(CurrentTracer.FrameBuffer.Resolution[0], CurrentTracer.FrameBuffer.Resolution[1]);
//	FilterGaussian(CurrentTracer.FrameBuffer.CudaFrameEstimate.GetPtr(), CurrentTracer.FrameBuffer.CudaFrameEstimateTemp.GetPtr(), CurrentTracer.FrameBuffer.Resolution[0], CurrentTracer.FrameBuffer.Resolution[1]);
	ComputeEstimate(CurrentTracer.FrameBuffer.Resolution[0], CurrentTracer.FrameBuffer.Resolution[1]);
	ToneMap(CurrentTracer.FrameBuffer.Resolution[0], CurrentTracer.FrameBuffer.Resolution[1]);

	CUDA::ThreadSynchronize();

	CurrentTracer.NoIterations++;

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void GetEstimate(int TracerID, unsigned char* pData)
{
	return;

	EDIT_TRACER(TracerID)
	CUDA::MemCopyDeviceToHost(CurrentTracer.FrameBuffer.CudaDisplayEstimate.GetPtr(), (ColorRGBAuc*)pData, CurrentTracer.FrameBuffer.CudaDisplayEstimate.GetNoElements());
}

EXPOSURE_RENDER_DLL void GetAutoFocusDistance(int TracerID, int FilmU, int FilmV, float& AutoFocusDistance)
{
	return;
	EDIT_TRACER(TracerID)
	ComputeAutoFocusDistance(FilmU, FilmV, AutoFocusDistance);
}

EXPOSURE_RENDER_DLL void GetNoIterations(int TracerID, int& NoIterations)
{
	return;
	EDIT_TRACER(TracerID)
	NoIterations = CurrentTracer.NoIterations; 
}

}
