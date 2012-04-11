/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Core.cuh"

DEVICE int* gpTracer = NULL;

typedef ExposureRender::List<ExposureRender::Volume, MAX_NO_VOLUMES>					Volumes;
typedef ExposureRender::List<ExposureRender::Light, MAX_NO_LIGHTS>						Lights;
typedef ExposureRender::List<ExposureRender::Object, MAX_NO_OBJECTS>					Objects;
typedef ExposureRender::List<ExposureRender::ClippingObject, MAX_NO_CLIPPING_OBJECTS>	ClippingObjects;
typedef ExposureRender::List<ExposureRender::Texture, MAX_NO_TEXTURES>					Textures;

DEVICE Volumes*			gpVolumes			= NULL;
DEVICE Lights*			gpLights			= NULL;
DEVICE Objects*			gpObjects			= NULL;
DEVICE ClippingObjects*	gpClippingObjects	= NULL;
DEVICE Textures*		gpTextures			= NULL;


#include "Tracer.cuh"


#include "Utilities.cuh"
/*

#include "GaussianFilter.cuh"
#include "BilateralFilter.cuh"
#include "MedianFilter.cuh"
#include "Estimate.cuh"

#include "SingleScattering.cuh"
#include "ToneMap.cuh"
#include "GradientMagnitude.cuh"
#include "AutoFocus.cuh"
*/

namespace ExposureRender
{

CudaList<Volume, MAX_NO_VOLUMES>					gVolumes("gpVolumes");
CudaList<Light, MAX_NO_LIGHTS>						gLights("gpLights");
CudaList<Object, MAX_NO_OBJECTS>					gObjects("gpObjects");
CudaList<ClippingObject, MAX_NO_CLIPPING_OBJECTS>	gClippingObjects("gpClippingObjects");
CudaList<Texture, MAX_NO_TEXTURES>					gTextures("gpTextures");

std::map<int, Tracer> gTracers;

EXPOSURE_RENDER_DLL void Resize(int TracerID, int Size[2])
{
	gTracers[TracerID].FrameBuffer.Resize(Resolution2i(Size));
//	gTracers.Synchronize();
}

EXPOSURE_RENDER_DLL void Reset(int TracerID)
{
	gTracers[TracerID].FrameBuffer.Reset();
	gTracers[TracerID].NoIterations = 0;
//	gTracers.Synchronize();
}

EXPOSURE_RENDER_DLL void InitializeTracer(int& ID)
{
//	cudaSetDevice(0);

//	gTracers.Bind(Tracer(), ID);
}

EXPOSURE_RENDER_DLL void DeinitializeTracer(int ID)
{
//	gTracers.Unbind(ID);
}

EXPOSURE_RENDER_DLL void BindVolume(Volume V, int& ID)
{
//	gSharedVolumes.Bind(Volume(V), ID);
}

EXPOSURE_RENDER_DLL void UnbindVolume(int ID)
{
//	gSharedVolumes.Unbind(ID);
}

EXPOSURE_RENDER_DLL void BindLight(Light L, int& ID)
{
}

EXPOSURE_RENDER_DLL void UnbindLight(int ID)
{
//	gSharedLights.Unbind(ID);
}

EXPOSURE_RENDER_DLL void BindObject(Object O, int& ID)
{
}

EXPOSURE_RENDER_DLL void UnbindObject(int ID)
{
//	gSharedObjects.Unbind(ID);
}

EXPOSURE_RENDER_DLL void BindClippingObject(ClippingObject C, int& ID)
{
}

EXPOSURE_RENDER_DLL void UnbindClippingObject(int ID)
{
//	gSharedClippingObjects.Unbind(ID);
}

EXPOSURE_RENDER_DLL void BindTexture(Texture Texture, int& ID)
{
}

EXPOSURE_RENDER_DLL void UnbindTexture(int ID)
{
//	gSharedTextures.Unbind(ID);
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

EXPOSURE_RENDER_DLL void BindOpacity1D(int TracerID, ScalarTransferFunction1D Opacity1D)
{
	gTracers[TracerID].Opacity1D = Opacity1D;
//	gTracers.Synchronize();
}

EXPOSURE_RENDER_DLL void BindDiffuse1D(int TracerID, ColorTransferFunction1D Diffuse1D)
{
	gTracers[TracerID].Diffuse1D = Diffuse1D;
//	gTracers.Synchronize();
}

EXPOSURE_RENDER_DLL void BindSpecular1D(int TracerID, ColorTransferFunction1D Specular1D)
{
	gTracers[TracerID].Specular1D = Specular1D;
//	gTracers.Synchronize();
}

EXPOSURE_RENDER_DLL void BindGlossiness1D(int TracerID, ScalarTransferFunction1D Glossiness1D)
{
	gTracers[TracerID].Glossiness1D = Glossiness1D;
//	gTracers.Synchronize();
}

EXPOSURE_RENDER_DLL void BindEmission1D(int TracerID, ColorTransferFunction1D Emission1D)
{
	gTracers[TracerID].Emission1D = Emission1D;
//	gTracers.Synchronize();
}

EXPOSURE_RENDER_DLL void BindCamera(int TracerID, Camera Camera)
{
	gTracers[TracerID].Camera = Camera;
//	gTracers.Synchronize();
}

EXPOSURE_RENDER_DLL void BindRenderSettings(int TracerID, RenderSettings RenderSettings)
{
	gTracers[TracerID].RenderSettings = RenderSettings;
//	gTracers.Synchronize();
}

EXPOSURE_RENDER_DLL void RenderEstimate(int TracerID)
{
//	CUDA::HostToConstantDevice(&TracerID, gActiveTracerID);

	CUDA::ThreadSynchronize();

//	SingleScattering(gTracers[TracerID].FrameBuffer.Resolution[0], gTracers[TracerID].FrameBuffer.Resolution[1]);
//	ComputeEstimate(gTracers[TracerID].FrameBuffer.Resolution[0], gTracers[TracerID].FrameBuffer.Resolution[1]);
//	ToneMap(gTracers[TracerID].FrameBuffer.Resolution[0], gTracers[TracerID].FrameBuffer.Resolution[1]);

	CUDA::ThreadSynchronize();

	gTracers[TracerID].NoIterations++; 
//	gTracers.Synchronize();
}

EXPOSURE_RENDER_DLL void GetEstimate(int TracerID, unsigned char* pData)
{
	CUDA::MemCopyDeviceToHost(gTracers[TracerID].FrameBuffer.CudaDisplayEstimate.GetPtr(), (ColorRGBAuc*)pData, gTracers[TracerID].FrameBuffer.CudaDisplayEstimate.GetNoElements());
}

EXPOSURE_RENDER_DLL void GetAutoFocusDistance(int TracerID, int FilmU, int FilmV, float& AutoFocusDistance)
{
	return;
//	ComputeAutoFocusDistance(FilmU, FilmV, AutoFocusDistance);
}

EXPOSURE_RENDER_DLL void GetNoIterations(int TracerID, int& NoIterations)
{
	NoIterations = gTracers[TracerID].NoIterations; 
}

}
