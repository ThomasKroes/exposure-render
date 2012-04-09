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

ExposureRender::Tracer gTracer;

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


static std::map<int, ExposureRender::Tracer>			gTracers;

ExposureRender::Tracer* gpCurrentTracer = NULL;

namespace ExposureRender
{

EXPOSURE_RENDER_DLL void Resize(int TracerID, int Size[2])
{
	gTracer.FrameBuffer.Resize(Resolution2i(Size));
}

EXPOSURE_RENDER_DLL void Reset(int TracerID)
{
	gTracer.FrameBuffer.Reset();
	gTracer.NoIterations = 0;
}

void SetTracer()
{
	if (gpCurrentTracer == NULL)
		cudaMalloc(&gpCurrentTracer, sizeof(Tracer));

	cudaMemcpy(gpCurrentTracer, &gTracer, sizeof(Tracer), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(gpTracer, &gpCurrentTracer, sizeof(gpCurrentTracer));

//	cudaFree(pTracer);
}

EXPOSURE_RENDER_DLL void BindOpacity1D(int TracerID, ErScalarTransferFunction1D Opacity1D)
{
	gTracer.Opacity1D = Opacity1D;
	SetTracer();
}

EXPOSURE_RENDER_DLL void BindDiffuse1D(int TracerID, ErColorTransferFunction1D Diffuse1D)
{
	gTracer.Diffuse1D = Diffuse1D;
	SetTracer();
}

EXPOSURE_RENDER_DLL void BindSpecular1D(int TracerID, ErColorTransferFunction1D Specular1D)
{
	gTracer.Specular1D = Specular1D;
	SetTracer();
}

EXPOSURE_RENDER_DLL void BindGlossiness1D(int TracerID, ErScalarTransferFunction1D Glossiness1D)
{
	gTracer.Glossiness1D = Glossiness1D;
	SetTracer();
}

EXPOSURE_RENDER_DLL void BindEmission1D(int TracerID, ErColorTransferFunction1D Emission1D)
{
	gTracer.Emission1D = Emission1D;
	SetTracer();
}

EXPOSURE_RENDER_DLL void BindCamera(int TracerID, ErCamera Camera)
{
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

	gTracer.Camera = Camera;

	SetTracer();
}

EXPOSURE_RENDER_DLL void BindVolume(int TracerID, int Resolution[3], float Spacing[3], unsigned short* pVoxels, bool NormalizeSize)
{
	gTracer.Volume.Set(Vec3f(Resolution[0], Resolution[1], Resolution[2]), Vec3f(Spacing[0], Spacing[1], Spacing[2]), pVoxels, NormalizeSize);
	SetTracer();
}

EXPOSURE_RENDER_DLL void BindLight(int TracerID, ErLight Light)
{
	std::map<int, ExposureRender::ErLight>::iterator It;

	It = gTracer.LightsMap.find(Light.ID);

	const bool Exists = It != gTracer.LightsMap.end();

	gTracer.LightsMap[Light.ID] = Light;

	ErShape& Shape = gTracer.LightsMap[Light.ID].Shape;

	switch (Shape.Type)
	{
		case Enums::Plane:		Shape.Area = PlaneArea(Vec2f(Shape.Size[0], Shape.Size[1]));				break;
		case Enums::Disk:		Shape.Area = DiskArea(Shape.OuterRadius);									break;
		case Enums::Ring:		Shape.Area = RingArea(Shape.OuterRadius, Shape.InnerRadius);				break;
		case Enums::Box:		Shape.Area = BoxArea(Vec3f(Shape.Size[0], Shape.Size[1], Shape.Size[2]));	break;
		case Enums::Sphere:		Shape.Area = SphereArea(Shape.OuterRadius);									break;
		case Enums::Cylinder:	Shape.Area = CylinderArea(Shape.OuterRadius, Shape.Size[2]);				break;
	}

	gTracer.CopyLights();
	
	SetTracer();
}

EXPOSURE_RENDER_DLL void UnbindLight(int TracerID, ErLight Light)
{
	std::map<int, ExposureRender::ErLight>::iterator It;

	It = gTracer.LightsMap.find(Light.ID);

	const bool Exists = It != gTracer.LightsMap.end();

	if (!Exists)
		return;

	gTracer.LightsMap.erase(It);
	
	gTracer.CopyLights();

	SetTracer();
}

EXPOSURE_RENDER_DLL void BindObject(int TracerID, ErObject Object)
{
	std::map<int, ExposureRender::ErObject>::iterator It;

	It = gTracer.ObjectsMap.find(Object.ID);

	gTracer.ObjectsMap[Object.ID] = Object;
	gTracer.CopyObjects();

	SetTracer();
}

EXPOSURE_RENDER_DLL void UnbindObject(int TracerID, ErObject Object)
{
	std::map<int, ExposureRender::ErObject>::iterator It;

	It = gTracer.ObjectsMap.find(Object.ID);

	if (It == gTracer.ObjectsMap.end())
		return;

	gTracer.ObjectsMap.erase(It);

	gTracer.CopyObjects();

	SetTracer();
}

EXPOSURE_RENDER_DLL void BindClippingObject(int TracerID, ErClippingObject ClippingObject)
{
	std::map<int, ExposureRender::ErClippingObject>::iterator It;

	It = gTracer.ClippingObjectsMap.find(ClippingObject.ID);

	gTracer.ClippingObjectsMap[ClippingObject.ID] = ClippingObject;

	gTracer.CopyClippingObjects();

	SetTracer();
}

EXPOSURE_RENDER_DLL void UnbindClippingObject(int TracerID, ErClippingObject ClippingObject)
{
	std::map<int, ExposureRender::ErClippingObject>::iterator It;

	It = gTracer.ClippingObjectsMap.find(ClippingObject.ID);

	if (It == gTracer.ClippingObjectsMap.end())
		return;

	gTracer.ClippingObjectsMap.erase(It);

	gTracer.CopyClippingObjects();

	SetTracer();
}

EXPOSURE_RENDER_DLL void BindTexture(int TracerID, ErTexture Texture)
{
	std::map<int, ExposureRender::ErTexture>::iterator It;

	It = gTracer.TexturesMap.find(Texture.ID);

	const bool Exists = It != gTracer.TexturesMap.end();

	gTracer.TexturesMap[Texture.ID] = Texture;

	if (Texture.Image.Dirty)
	{
		if (gTracer.TexturesMap[Texture.ID].Image.pData)
			CUDA::Free(gTracer.TexturesMap[Texture.ID].Image.pData);

		if (Texture.Image.pData)
		{
			const int NoPixels = gTracer.TexturesMap[Texture.ID].Image.Size[0] * gTracer.TexturesMap[Texture.ID].Image.Size[1];
		
			CUDA::Allocate(gTracer.TexturesMap[Texture.ID].Image.pData, NoPixels);
			CUDA::MemCopyHostToDevice(Texture.Image.pData, gTracer.TexturesMap[Texture.ID].Image.pData, NoPixels);
		}
	} 

	gTracer.CopyTextures();

	SetTracer();
}

EXPOSURE_RENDER_DLL void UnbindTexture(int TracerID, ErTexture Texture)
{
	std::map<int, ExposureRender::ErTexture>::iterator It;

	It = gTracer.TexturesMap.find(Texture.ID);

	if (It == gTracer.TexturesMap.end())
		return;

	if (It->second.Image.pData)
		CUDA::Free(It->second.Image.pData);

	gTracer.TexturesMap.erase(It);
	gTracer.CopyTextures();

	SetTracer();
}

EXPOSURE_RENDER_DLL void BindRenderSettings(int TracerID, ErRenderSettings RenderSettings)
{
	gTracer.RenderSettings = RenderSettings;
	SetTracer();
}

EXPOSURE_RENDER_DLL void BindFiltering(int TracerID, ErFiltering Filtering)
{
	// Frame estimate filter
	GaussianFilter Gaussian;
	
	Gaussian.KernelRadius = Filtering.FrameEstimateFilter.KernelRadius;

	const int KernelSize = (2 * Gaussian.KernelRadius) + 1;

	for (int i = 0; i < KernelSize; i++)
		Gaussian.KernelD[i] = Gauss2D(Filtering.FrameEstimateFilter.Sigma, Gaussian.KernelRadius - i, 0);

	gTracer.FrameEstimateFilter = Gaussian;
	
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

	gTracer.PostProcessingFilter = Bilateral;

	SetTracer();
}

EXPOSURE_RENDER_DLL void RenderEstimate(int TracerID)
{
	gKernelTimings.Reset();

	CUDA::ThreadSynchronize();

	SingleScattering();
//	FilterGaussian(gTracer.FrameBuffer.CudaFrameEstimate.GetPtr(), gTracer.FrameBuffer.CudaFrameEstimateTemp.GetPtr(), gTracer.FrameBuffer.Resolution[0], gTracer.FrameBuffer.Resolution[1]);
	ComputeEstimate();
	ToneMap();
//	FilterBilateral(gFrameBuffer.CudaDisplayEstimate.GetPtr(), gFrameBuffer.CudaDisplayEstimateTemp.GetPtr(), gFrameBuffer.CudaDisplayEstimateFiltered.GetPtr(), gFrameBuffer.Resolution[0], gFrameBuffer.Resolution[1]);
//	MedianFilter(gFrameBuffer.CudaDisplayEstimate.GetPtr(), gFrameBuffer.CudaDisplayEstimateFiltered.GetPtr(), gFrameBuffer.Resolution[0], gFrameBuffer.Resolution[1]);
//	Blend(gFrameBuffer.CudaDisplayEstimate.GetPtr(), gFrameBuffer.CudaDisplayEstimateFiltered.GetPtr(), gFrameBuffer.Resolution[0], gFrameBuffer.Resolution[1]);

	CUDA::ThreadSynchronize();

	gTracer.NoIterations++;

	SetTracer();
}

EXPOSURE_RENDER_DLL void GetEstimate(int TracerID, unsigned char* pData)
{
	CUDA::MemCopyDeviceToHost(gTracer.FrameBuffer.CudaDisplayEstimate.GetPtr(), (ColorRGBAuc*)pData, gTracer.FrameBuffer.CudaDisplayEstimate.GetNoElements());
}

EXPOSURE_RENDER_DLL void GetAutoFocusDistance(int TracerID, int FilmU, int FilmV, float& AutoFocusDistance)
{
	ComputeAutoFocusDistance(FilmU, FilmV, AutoFocusDistance);
}

EXPOSURE_RENDER_DLL void GetKernelTimings(int TracerID, ErKernelTimings& KernelTimings)
{
	KernelTimings = gKernelTimings;
}

EXPOSURE_RENDER_DLL void GetNoIterations(int TracerID, int& NoIterations)
{
	NoIterations = gTracer.NoIterations; 
}

EXPOSURE_RENDER_DLL void DeinitializeTracer(int TracerID)
{
	gTracer.FrameBuffer.Free();
}

EXPOSURE_RENDER_DLL void InitializeTracer(int& TracerID)
{
//	DeinitializeTracer(DeinitializeTracer);
}

}