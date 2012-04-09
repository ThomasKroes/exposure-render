/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <thrust/reduce.h>

#include "General.cuh"

ExposureRender::ErKernelTimings gKernelTimings;

#include "Core.cuh"
#include "CudaUtilities.cuh"
#include "Framebuffer.cuh"
#include "Benchmark.cuh"
#include "Filter.cuh"


__device__ int* gpTracer;
__device__ int*	gpSharedResources = NULL;


int	gNoIterations = 0;

#include "Tracer.cuh"

#include "SharedResources.cuh"

ExposureRender::SharedResources* pSharedResources = NULL;

ExposureRender::SharedResources gSharedResources;

#include "GaussianFilter.cuh"
#include "BilateralFilter.cuh"
#include "MedianFilter.cuh"
#include "Estimate.cuh"
#include "Utilities.cuh"
#include "SingleScattering.cuh"
#include "Metropolis.cuh"
#include "ToneMap.cuh"
#include "Blend.cuh"
#include "GradientMagnitude.cuh"
#include "AutoFocus.cuh"


static std::map<int, ExposureRender::Tracer> gTracers;
int gNoTracers = 0;

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
	TracerID = gNoTracers;
	gTracers[gNoTracers] = Tracer();
	gNoTracers++;

	EDIT_TRACER(TracerID)

	CurrentTracer.FrameBuffer.Free();

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void DeinitializeTracer(int TracerID)
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

	const Vec3f N = Normalize(ToVec3f(Camera.Target) - ToVec3f(Camera.Pos));
	const Vec3f U = Normalize(Cross(N, ToVec3f(Camera.Up)));
	const Vec3f V = Normalize(Cross(N, U));

	Camera.N[0] = N[0];
	Camera.N[1] = N[1];
	Camera.N[2] = N[2];
	Camera.U[0] = U[0];
	Camera.U[1] = U[1];
	Camera.U[2] = U[2];
	Camera.V[0] = V[0];
	Camera.V[1] = V[1];
	Camera.V[2] = V[2];

	if (Camera.FocalDistance == -1.0f)
		Camera.FocalDistance = (ToVec3f(Camera.Target) - ToVec3f(Camera.Pos)).Length();

	float Scale = 0.0f;

	Scale = tanf((0.5f * Camera.FOV / RAD_F));

	const float AspectRatio = (float)Camera.FilmHeight / (float)Camera.FilmWidth;

	if (AspectRatio > 1.0f)
	{
		Camera.Screen[0][0] = -Scale;
		Camera.Screen[0][1] = Scale;
		Camera.Screen[1][0] = -Scale * AspectRatio;
		Camera.Screen[1][1] = Scale * AspectRatio;
	}
	else
	{
		Camera.Screen[0][0] = -Scale / AspectRatio;
		Camera.Screen[0][1] = Scale / AspectRatio;
		Camera.Screen[1][0] = -Scale;
		Camera.Screen[1][1] = Scale;
	}

	Camera.InvScreen[0] = (Camera.Screen[0][1] - Camera.Screen[0][0]) / (float)Camera.FilmWidth;
	Camera.InvScreen[1] = (Camera.Screen[1][1] - Camera.Screen[1][0]) / (float)Camera.FilmHeight;

	CurrentTracer.Camera = Camera;

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void BindVolume(ErVolume& Volume)
{
}

EXPOSURE_RENDER_DLL void UnbindVolume(ErVolume& Volume)
{
}

EXPOSURE_RENDER_DLL void BindLight(ErLight Light)
{
	EDIT_TRACER(TracerID)

	std::map<int, ExposureRender::ErLight>::iterator It;

	It = CurrentTracer.LightsMap.find(Light.ID);

	const bool Exists = It != CurrentTracer.LightsMap.end();

	CurrentTracer.LightsMap[Light.ID] = Light;

	ErShape& Shape = CurrentTracer.LightsMap[Light.ID].Shape;

	switch (Shape.Type)
	{
		case Enums::Plane:		Shape.Area = PlaneArea(Vec2f(Shape.Size[0], Shape.Size[1]));				break;
		case Enums::Disk:		Shape.Area = DiskArea(Shape.OuterRadius);									break;
		case Enums::Ring:		Shape.Area = RingArea(Shape.OuterRadius, Shape.InnerRadius);				break;
		case Enums::Box:		Shape.Area = BoxArea(Vec3f(Shape.Size[0], Shape.Size[1], Shape.Size[2]));	break;
		case Enums::Sphere:		Shape.Area = SphereArea(Shape.OuterRadius);									break;
		case Enums::Cylinder:	Shape.Area = CylinderArea(Shape.OuterRadius, Shape.Size[2]);				break;
	}

	CurrentTracer.CopyLights();
	
	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void UnbindLight(ErLight Light)
{
	EDIT_TRACER(TracerID)

	std::map<int, ExposureRender::ErLight>::iterator It;

	It = CurrentTracer.LightsMap.find(Light.ID);

	const bool Exists = It != CurrentTracer.LightsMap.end();

	if (!Exists)
		return;

	CurrentTracer.LightsMap.erase(It);
	
	CurrentTracer.CopyLights();

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void BindObject(ErObject Object)
{
	EDIT_TRACER(TracerID)

	std::map<int, ExposureRender::ErObject>::iterator It;

	It = CurrentTracer.ObjectsMap.find(Object.ID);

	CurrentTracer.ObjectsMap[Object.ID] = Object;
	CurrentTracer.CopyObjects();

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void UnbindObject(ErObject Object)
{
	EDIT_TRACER(TracerID)

	std::map<int, ExposureRender::ErObject>::iterator It;

	It = CurrentTracer.ObjectsMap.find(Object.ID);

	if (It == CurrentTracer.ObjectsMap.end())
		return;

	CurrentTracer.ObjectsMap.erase(It);

	CurrentTracer.CopyObjects();

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void BindClippingObject(ErClippingObject ClippingObject)
{
	EDIT_TRACER(TracerID)

	std::map<int, ExposureRender::ErClippingObject>::iterator It;

	It = CurrentTracer.ClippingObjectsMap.find(ClippingObject.ID);

	CurrentTracer.ClippingObjectsMap[ClippingObject.ID] = ClippingObject;

	CurrentTracer.CopyClippingObjects();

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void UnbindClippingObject(ErClippingObject ClippingObject)
{
	EDIT_TRACER(TracerID)

	std::map<int, ExposureRender::ErClippingObject>::iterator It;

	It = CurrentTracer.ClippingObjectsMap.find(ClippingObject.ID);

	if (It == CurrentTracer.ClippingObjectsMap.end())
		return;

	CurrentTracer.ClippingObjectsMap.erase(It);

	CurrentTracer.CopyClippingObjects();

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void BindTexture(ErTexture Texture)
{
	EDIT_TRACER(TracerID)

	std::map<int, ExposureRender::ErTexture>::iterator It;

	It = CurrentTracer.TexturesMap.find(Texture.ID);

	const bool Exists = It != CurrentTracer.TexturesMap.end();

	CurrentTracer.TexturesMap[Texture.ID] = Texture;

	if (Texture.Image.Dirty)
	{
		if (CurrentTracer.TexturesMap[Texture.ID].Image.pData)
			CUDA::Free(CurrentTracer.TexturesMap[Texture.ID].Image.pData);

		if (Texture.Image.pData)
		{
			const int NoPixels = CurrentTracer.TexturesMap[Texture.ID].Image.Size[0] * CurrentTracer.TexturesMap[Texture.ID].Image.Size[1];
		
			CUDA::Allocate(CurrentTracer.TexturesMap[Texture.ID].Image.pData, NoPixels);
			CUDA::MemCopyHostToDevice(Texture.Image.pData, CurrentTracer.TexturesMap[Texture.ID].Image.pData, NoPixels);
		}
	} 

	CurrentTracer.CopyTextures();

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void UnbindTexture(ErTexture Texture)
{
	EDIT_TRACER(TracerID)

	std::map<int, ExposureRender::ErTexture>::iterator It;

	It = CurrentTracer.TexturesMap.find(Texture.ID);

	if (It == CurrentTracer.TexturesMap.end())
		return;

	if (It->second.Image.pData)
		CUDA::Free(It->second.Image.pData);

	CurrentTracer.TexturesMap.erase(It);
	CurrentTracer.CopyTextures();

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
	EDIT_TRACER(TracerID)

	gKernelTimings.Reset();

	CUDA::ThreadSynchronize();

	SingleScattering(CurrentTracer.FrameBuffer.Resolution[0], CurrentTracer.FrameBuffer.Resolution[1]);
//	FilterGaussian(CurrentTracer.FrameBuffer.CudaFrameEstimate.GetPtr(), CurrentTracer.FrameBuffer.CudaFrameEstimateTemp.GetPtr(), CurrentTracer.FrameBuffer.Resolution[0], CurrentTracer.FrameBuffer.Resolution[1]);
	ComputeEstimate(CurrentTracer.FrameBuffer.Resolution[0], CurrentTracer.FrameBuffer.Resolution[1]);
	ToneMap(CurrentTracer.FrameBuffer.Resolution[0], CurrentTracer.FrameBuffer.Resolution[1]);
//	FilterBilateral(gFrameBuffer.CudaDisplayEstimate.GetPtr(), gFrameBuffer.CudaDisplayEstimateTemp.GetPtr(), gFrameBuffer.CudaDisplayEstimateFiltered.GetPtr(), gFrameBuffer.Resolution[0], gFrameBuffer.Resolution[1]);
//	MedianFilter(gFrameBuffer.CudaDisplayEstimate.GetPtr(), gFrameBuffer.CudaDisplayEstimateFiltered.GetPtr(), gFrameBuffer.Resolution[0], gFrameBuffer.Resolution[1]);
//	Blend(gFrameBuffer.CudaDisplayEstimate.GetPtr(), gFrameBuffer.CudaDisplayEstimateFiltered.GetPtr(), gFrameBuffer.Resolution[0], gFrameBuffer.Resolution[1]);

	CUDA::ThreadSynchronize();

	CurrentTracer.NoIterations++;

	BindTracer(TracerID);
}

EXPOSURE_RENDER_DLL void GetEstimate(int TracerID, unsigned char* pData)
{
	EDIT_TRACER(TracerID)
	CUDA::MemCopyDeviceToHost(CurrentTracer.FrameBuffer.CudaDisplayEstimate.GetPtr(), (ColorRGBAuc*)pData, CurrentTracer.FrameBuffer.CudaDisplayEstimate.GetNoElements());
}

EXPOSURE_RENDER_DLL void GetAutoFocusDistance(int TracerID, int FilmU, int FilmV, float& AutoFocusDistance)
{
	EDIT_TRACER(TracerID)
	ComputeAutoFocusDistance(FilmU, FilmV, AutoFocusDistance);
}

EXPOSURE_RENDER_DLL void GetKernelTimings(int TracerID, ErKernelTimings& KernelTimings)
{
	EDIT_TRACER(TracerID)
	KernelTimings = gKernelTimings;
}

EXPOSURE_RENDER_DLL void GetNoIterations(int TracerID, int& NoIterations)
{
	EDIT_TRACER(TracerID)
	NoIterations = CurrentTracer.NoIterations; 
}

}
