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

#include "Core.cuh"
#include "CudaUtilities.cuh"
#include "General.cuh"
#include "Framebuffer.cuh"
#include "Benchmark.cuh"
#include "Filter.cuh"

texture<unsigned short, cudaTextureType3D, cudaReadModeNormalizedFloat>		gTexIntensity;
texture<unsigned short, cudaTextureType3D, cudaReadModeNormalizedFloat>		gTexExtinction;
texture<float, cudaTextureType1D, cudaReadModeElementType>					gTexOpacity;
texture<float4, cudaTextureType1D, cudaReadModeElementType>					gTexDiffuse;
texture<float4, cudaTextureType1D, cudaReadModeElementType>					gTexSpecular;
texture<float, cudaTextureType1D, cudaReadModeElementType>					gTexGlossiness;
texture<float4, cudaTextureType1D, cudaReadModeElementType>					gTexEmission;

cudaChannelFormatDesc gFloatChannelDesc		= cudaCreateChannelDesc<float>();
cudaChannelFormatDesc gFloat4ChannelDesc	= cudaCreateChannelDesc<float4>();

cudaArray* gpIntensity	= NULL;
cudaArray* gpExtinction	= NULL;
cudaArray* gpOpacity	= NULL;
cudaArray* gpDiffuse	= NULL;
cudaArray* gpSpecular	= NULL;
cudaArray* gpGlossiness	= NULL;
cudaArray* gpEmission	= NULL;

CD ErVolume			gVolume;
CD ErCamera			gCamera;
CD ErLights			gLights;
CD ErClippers		gClippers;
CD ErReflectors		gReflectors;
CD ErDenoise		gDenoise;
CD ErScattering		gScattering;
CD ErRange			gOpacityRange;
CD ErRange			gDiffuseRange;
CD ErRange			gSpecularRange;
CD ErRange			gGlossinessRange;
CD ErRange			gEmissionRange;
CD GaussianFilter	gFrameEstimateFilter;
CD BilateralFilter	gPostProcessingFilter;

FrameBuffer gFrameBuffer;

#include "GaussianFilter.cuh"
#include "BilateralFilter.cuh"
#include "Denoise.cuh"
#include "Estimate.cuh"
#include "Utilities.cuh"
#include "SingleScattering.cuh"
#include "Metropolis.cuh"
#include "ToneMap.cuh"
#include "Blend.cuh"
#include "GradientMagnitude.cuh"
#include "AutoFocus.cuh"

void ErResize(int Size[2])
{
	gFrameBuffer.Resize(Resolution2i(Size));
}

void ErResetFrameBuffer()
{
	gFrameBuffer.Reset();
}

void ErUnbindDensityBuffer(void)
{
	CUDA::FreeArray(gpIntensity);
	CUDA::UnbindTexture(gTexIntensity);
}

void ErBindIntensityBuffer(unsigned short* pBuffer, int Extent[3])
{
	ErUnbindDensityBuffer();
	CUDA::BindTexture3D(gTexIntensity, Extent, pBuffer, gpIntensity);
}

void ErBindOpacity1D(float Opacity[NO_GRADIENT_STEPS], float Range[2])
{
	ErRange Int;
	Int.Set(Range);

	CUDA::HostToConstantDevice(&Int, "gOpacityRange");

	gTexOpacity.normalized		= true;
	gTexOpacity.filterMode		= cudaFilterModeLinear;
	gTexOpacity.addressMode[0]	= cudaAddressModeClamp;

	if (gpOpacity == NULL)
		cudaMallocArray(&gpOpacity, &gFloatChannelDesc, NO_GRADIENT_STEPS, 1);

	cudaMemcpyToArray(gpOpacity, 0, 0, Opacity, NO_GRADIENT_STEPS * sizeof(float), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexOpacity, gpOpacity, gFloatChannelDesc);
}

void ErBindDiffuse1D(float Diffuse[3][NO_GRADIENT_STEPS], float Range[2])
{
	ErRange Int;
	Int.Set(Range);

	CUDA::HostToConstantDevice(&Int, "gDiffuseRange");

	gTexDiffuse.normalized		= true;
	gTexDiffuse.filterMode		= cudaFilterModeLinear;
	gTexDiffuse.addressMode[0]	= cudaAddressModeClamp;

	if (gpDiffuse == NULL)
		cudaMallocArray(&gpDiffuse, &gFloat4ChannelDesc, NO_GRADIENT_STEPS, 1);

	ColorXYZAf* pDiffuseXYZA = new ColorXYZAf[NO_GRADIENT_STEPS];

	for (int i = 0; i < NO_GRADIENT_STEPS; i++)
		pDiffuseXYZA[i].FromRGB(Diffuse[0][i], Diffuse[1][i], Diffuse[2][i]);

	cudaMemcpyToArray(gpDiffuse, 0, 0, pDiffuseXYZA, NO_GRADIENT_STEPS * sizeof(float4), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexDiffuse, gpDiffuse, gFloat4ChannelDesc);

	delete[] pDiffuseXYZA;
}

void ErBindSpecular1D(float Specular[3][NO_GRADIENT_STEPS], float Range[2])
{
	ErRange Int;
	Int.Set(Range);

	CUDA::HostToConstantDevice(&Int, "gSpecularRange");

	gTexSpecular.normalized		= true;
	gTexSpecular.filterMode		= cudaFilterModeLinear;
	gTexSpecular.addressMode[0]	= cudaAddressModeClamp;

	if (gpSpecular == NULL)
		cudaMallocArray(&gpSpecular, &gFloat4ChannelDesc, NO_GRADIENT_STEPS, 1);

	ColorXYZAf* pSpecularXYZA = new ColorXYZAf[NO_GRADIENT_STEPS];

	for (int i = 0; i < NO_GRADIENT_STEPS; i++)
		pSpecularXYZA[i].FromRGB(Specular[0][i], Specular[1][i], Specular[2][i]);

	cudaMemcpyToArray(gpSpecular, 0, 0, pSpecularXYZA, NO_GRADIENT_STEPS * sizeof(float4), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexSpecular, gpSpecular, gFloat4ChannelDesc);

	delete[] pSpecularXYZA;
}

void ErBindGlossiness1D(float Glossiness[NO_GRADIENT_STEPS], float Range[2])
{
	ErRange Int;
	Int.Set(Range);

	CUDA::HostToConstantDevice(&Int, "gGlossinessRange");

	gTexGlossiness.normalized		= true;
	gTexGlossiness.filterMode		= cudaFilterModeLinear;
	gTexGlossiness.addressMode[0]	= cudaAddressModeClamp;

	if (gpGlossiness == NULL)
		cudaMallocArray(&gpGlossiness, &gFloatChannelDesc, NO_GRADIENT_STEPS, 1);

	cudaMemcpyToArray(gpGlossiness, 0, 0, Glossiness, NO_GRADIENT_STEPS * sizeof(float),  cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexGlossiness, gpGlossiness, gFloatChannelDesc);
}

void ErBindEmission1D(float Emission[3][NO_GRADIENT_STEPS], float Range[2])
{
	ErRange Int;
	Int.Set(Range);

	CUDA::HostToConstantDevice(&Int, "gEmissionRange");

	gTexEmission.normalized		= true;
	gTexEmission.filterMode		= cudaFilterModeLinear;
	gTexEmission.addressMode[0]	= cudaAddressModeClamp;

	if (gpEmission == NULL)
		cudaMallocArray(&gpEmission, &gFloat4ChannelDesc, NO_GRADIENT_STEPS, 1);

	ColorXYZAf* pEmissionXYZA = new ColorXYZAf[NO_GRADIENT_STEPS];

	for (int i = 0; i < NO_GRADIENT_STEPS; i++)
		pEmissionXYZA[i].FromRGB(Emission[0][i], Emission[1][i], Emission[2][i]);

	cudaMemcpyToArray(gpEmission, 0, 0, pEmissionXYZA, NO_GRADIENT_STEPS * sizeof(float4),  cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexEmission, gpEmission, gFloat4ChannelDesc);

	delete[] pEmissionXYZA;
}

void ErUnbindOpacity1D(void)
{
	CUDA::FreeArray(gpOpacity);
	CUDA::UnbindTexture(gTexOpacity);
}

void ErUnbindDiffuse1D(void)
{
	CUDA::FreeArray(gpDiffuse);
	CUDA::UnbindTexture(gTexDiffuse);
}

void ErUnbindSpecular1D(void)
{
	CUDA::FreeArray(gpSpecular);
	CUDA::UnbindTexture(gTexSpecular);
}

void ErUnbindGlossiness1D(void)
{
	CUDA::FreeArray(gpGlossiness);
	CUDA::UnbindTexture(gTexGlossiness);
}

void ErUnbindEmission1D(void)
{
	CUDA::FreeArray(gpEmission);
	CUDA::UnbindTexture(gTexEmission);
}

void ErBindVolume(ErVolume* pVolume)
{
	CUDA::HostToConstantDevice(pVolume, "gVolume");
}

void ErBindCamera(ErCamera* pCamera)
{
	const Vec3f N = Normalize(ToVec3f(pCamera->Target) - ToVec3f(pCamera->Pos));
	const Vec3f U = Normalize(Cross(N, ToVec3f(pCamera->Up)));
	const Vec3f V = Normalize(Cross(N, U));

	pCamera->N[0] = N[0];
	pCamera->N[1] = N[1];
	pCamera->N[2] = N[2];
	pCamera->U[0] = U[0];
	pCamera->U[1] = U[1];
	pCamera->U[2] = U[2];
	pCamera->V[0] = V[0];
	pCamera->V[1] = V[1];
	pCamera->V[2] = V[2];

	if (pCamera->FocalDistance == -1.0f)
		pCamera->FocalDistance = (ToVec3f(pCamera->Target) - ToVec3f(pCamera->Pos)).Length();

	float Scale = 0.0f;

	Scale = tanf((0.5f * pCamera->FOV / RAD_F));

	const float AspectRatio = (float)pCamera->FilmHeight / (float)pCamera->FilmWidth;

	if (AspectRatio > 1.0f)
	{
		pCamera->Screen[0][0] = -Scale;
		pCamera->Screen[0][1] = Scale;
		pCamera->Screen[1][0] = -Scale * AspectRatio;
		pCamera->Screen[1][1] = Scale * AspectRatio;
	}
	else
	{
		pCamera->Screen[0][0] = -Scale / AspectRatio;
		pCamera->Screen[0][1] = Scale / AspectRatio;
		pCamera->Screen[1][0] = -Scale;
		pCamera->Screen[1][1] = Scale;
	}

	pCamera->InvScreen[0] = (pCamera->Screen[0][1] - pCamera->Screen[0][0]) / (float)pCamera->FilmWidth;
	pCamera->InvScreen[1] = (pCamera->Screen[1][1] - pCamera->Screen[1][0]) / (float)pCamera->FilmHeight;

	CUDA::HostToConstantDevice(pCamera, "gCamera");
}

void ErBindLights(ErLights* pLights)
{
	CUDA::HostToConstantDevice(pLights, "gLights");
}

void ErBindClippers(ErClippers* pClippers)
{
	CUDA::HostToConstantDevice(pClippers, "gClippers");
}

void ErBindReflectors(ErReflectors* pReflectors)
{
	CUDA::HostToConstantDevice(pReflectors, "gReflectors");
}

void ErBindDenoise(ErDenoise* pDenoise)
{
	CUDA::HostToConstantDevice(pDenoise, "gDenoise");
}

void ErBindScattering(ErScattering* pScattering)
{
	CUDA::HostToConstantDevice(pScattering, "gScattering");
}

void ErBindFiltering(ErFiltering* pFiltering)
{
	// Frame estimate filter
	GaussianFilter Gaussian;
	
	Gaussian.KernelRadius = pFiltering->FrameEstimateFilter.KernelRadius;

	const int KernelSize = (2 * Gaussian.KernelRadius) + 1;

	for (int i = 0; i < KernelSize; i++)
		Gaussian.KernelD[i] = Gauss2D(pFiltering->FrameEstimateFilter.Sigma, Gaussian.KernelRadius - i, 0);

	CUDA::HostToConstantDevice(&Gaussian, "gFrameEstimateFilter");

	// Post processing filter
	BilateralFilter Bilateral;

	const int SigmaMax = max(pFiltering->PostProcessingFilter.SigmaD, pFiltering->PostProcessingFilter.SigmaR);
	
	Bilateral.KernelRadius = ceilf(2.0f * (float)SigmaMax);  

	const float TwoSigmaRSquared = 2 * pFiltering->PostProcessingFilter.SigmaR * pFiltering->PostProcessingFilter.SigmaR;

	const int kernelSize = Bilateral.KernelRadius * 2 + 1;
	const int center = (kernelSize - 1) / 2;

	for (int x = -center; x < -center + kernelSize; x++)
		Bilateral.KernelD[x + center] = Gauss2D(pFiltering->PostProcessingFilter.SigmaD, x, 0.0f);

	for (int i = 0; i < 256; i++)
		Bilateral.GaussSimilarity[i] = expf(-((float)i / TwoSigmaRSquared));

	CUDA::HostToConstantDevice(&Bilateral, "gPostProcessingFilter");
}

void ErRenderEstimate()
{
	FrameBuffer* pDevFrameBuffer = NULL;
	cudaMalloc(&pDevFrameBuffer, sizeof(FrameBuffer));
	CUDA::MemCopyHostToDevice(&gFrameBuffer, pDevFrameBuffer);

	SingleScattering(pDevFrameBuffer, gFrameBuffer.Resolution[0], gFrameBuffer.Resolution[1]);
	FilterGaussian(gFrameBuffer.CudaFrameEstimate.GetPtr(), gFrameBuffer.CudaFrameEstimateTemp.GetPtr(), gFrameBuffer.Resolution[0], gFrameBuffer.Resolution[1]);
	ComputeEstimate(pDevFrameBuffer, gFrameBuffer.Resolution[0], gFrameBuffer.Resolution[1]);
	ToneMap(pDevFrameBuffer, gFrameBuffer.Resolution[0], gFrameBuffer.Resolution[1]);
	FilterBilateral(gFrameBuffer.CudaDisplayEstimate.GetPtr(), gFrameBuffer.CudaDisplayEstimateTemp.GetPtr(), gFrameBuffer.CudaDisplayEstimateFiltered.GetPtr(), gFrameBuffer.Resolution[0], gFrameBuffer.Resolution[1]);
	Blend(gFrameBuffer.CudaDisplayEstimate.GetPtr(), gFrameBuffer.CudaDisplayEstimateFiltered.GetPtr(), gFrameBuffer.Resolution[0], gFrameBuffer.Resolution[1]);

	cudaFree(pDevFrameBuffer);
}

void ErGetEstimate(unsigned char* pData)
{
	CUDA::MemCopyDeviceToHost(gFrameBuffer.CudaDisplayEstimate.GetPtr(), (ColorRGBAuc*)pData, gFrameBuffer.CudaDisplayEstimate.GetNoElements());
}

void ErRecordBenchmarkImage()
{
	CUDA::MemCopyDeviceToDevice(gFrameBuffer.CudaDisplayEstimate.GetPtr(), gFrameBuffer.BenchmarkEstimateRgbaLdr.GetPtr(), gFrameBuffer.CudaDisplayEstimate.GetNoElements());
}

void ErGetRunningVariance(float& RunningVariance)
{
	FrameBuffer* pDevFrameBuffer = NULL;
	cudaMalloc(&pDevFrameBuffer, sizeof(FrameBuffer));
	cudaMemcpy(pDevFrameBuffer, &gFrameBuffer, sizeof(FrameBuffer), cudaMemcpyHostToDevice);

	thrust::device_ptr<float> dev_ptr(gFrameBuffer.CudaVariance.GetPtr()); 

	float Sum = thrust::reduce(dev_ptr, dev_ptr + gFrameBuffer.Resolution[0] * gFrameBuffer.Resolution[1]);
	
	RunningVariance = (Sum / (float)(gFrameBuffer.Resolution[0] * gFrameBuffer.Resolution[1]));

	cudaFree(pDevFrameBuffer);
}

void ErGetAverageNrmsError(float& AverageNrmsError)
{
	FrameBuffer* pDevFrameBuffer = NULL;
	cudaMalloc(&pDevFrameBuffer, sizeof(FrameBuffer));
	CUDA::MemCopyHostToDevice(&gFrameBuffer, pDevFrameBuffer);

	ComputeAverageNrmsError(gFrameBuffer, pDevFrameBuffer, gFrameBuffer.Resolution[0], gFrameBuffer.Resolution[1], AverageNrmsError);

	cudaFree(pDevFrameBuffer);
}

void ErGetMaximumGradientMagnitude(float& MaximumGradientMagnitude, int Extent[3])
{
	ComputeGradientMagnitudeVolume(Extent, MaximumGradientMagnitude);
}

void ErGetAutoFocusDistance(int FilmU, int FilmV, float& AutoFocusDistance)
{
	ComputeAutoFocusDistance(FilmU, FilmV, AutoFocusDistance);
}

void ErDeinitialize();

void ErInitialize()
{
	ErDeinitialize();
}

void ErDeinitialize()
{
	ErUnbindDensityBuffer();
	ErUnbindOpacity1D();
	ErUnbindDiffuse1D();
	ErUnbindSpecular1D();
	ErUnbindGlossiness1D();
	ErUnbindEmission1D();
}