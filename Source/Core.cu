/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Core.cuh"

texture<short, cudaTextureType3D, cudaReadModeNormalizedFloat>		gTexDensity;
texture<short, cudaTextureType3D, cudaReadModeNormalizedFloat>		gTexGradientMagnitude;
texture<float, cudaTextureType3D, cudaReadModeElementType>			gTexExtinction;
texture<float, cudaTextureType1D, cudaReadModeElementType>			gTexOpacity;
texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexDiffuse;
texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexSpecular;
texture<float, cudaTextureType1D, cudaReadModeElementType>			gTexRoughness;
texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexEmission;
texture<float4, cudaTextureType2D, cudaReadModeElementType>			gTexRunningEstimateXyza;
texture<float4, cudaTextureType2D, cudaReadModeElementType>			gTexFrameEstimateXyza;
texture<float4, cudaTextureType2D, cudaReadModeElementType>			gTexFrameBlurXyza;
texture<float4, cudaTextureType2D, cudaReadModeElementType>			gTexRunningSpecularBloomXyza;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat>		gTexRunningEstimateRgba;

cudaArray* gpDensityArray				= NULL;
cudaArray* gpGradientMagnitudeArray		= NULL;
cudaArray* gpOpacityArray				= NULL;
cudaArray* gpDiffuseArray				= NULL;
cudaArray* gpSpecularArray				= NULL;
cudaArray* gpRoughnessArray				= NULL;
cudaArray* gpEmissionArray				= NULL;

CD float3		gAaBbMin;
CD float3		gAaBbMax;
CD float3		gInvAaBbMin;
CD float3		gInvAaBbMax;
CD float		gIntensityMin;
CD float		gIntensityMax;
CD float		gIntensityRange;
CD float		gIntensityInvRange;
CD float		gStepSize;
CD float		gStepSizeShadow;
CD float		gDensityScale;
CD float		gGradientDelta;
CD float		gInvGradientDelta;
CD float3		gGradientDeltaX;
CD float3		gGradientDeltaY;
CD float3		gGradientDeltaZ;
CD int			gFilmWidth;
CD int			gFilmHeight;
CD int			gFilmNoPixels;
CD int			gFilterWidth;
CD float		gFilterWeights[10];
CD float		gExposure;
CD float		gInvExposure;
CD float		gGamma;
CD float		gInvGamma;
CD float		gDenoiseEnabled;
CD float		gDenoiseWindowRadius;
CD float		gDenoiseInvWindowArea;
CD float		gDenoiseNoise;
CD float		gDenoiseWeightThreshold;
CD float		gDenoiseLerpThreshold;
CD float		gDenoiseLerpC;
CD float		gNoIterations;
CD float		gInvNoIterations;

#define TF_NO_SAMPLES		128
#define INV_TF_NO_SAMPLES	1.0f / (float)TF_NO_SAMPLES

#include "Model.cuh"
#include "View.cuh"
#include "Blur.cuh"
#include "Denoise.cuh"
#include "Estimate.cuh"
#include "Utilities.cuh"
#include "SingleScattering.cuh"
#include "NearestIntersection.cuh"
#include "SpecularBloom.cuh"
#include "ToneMap.cuh"

CCudaModel	gModel;
CCudaView	gRenderCanvasView;
CCudaView	gNavigatorView;

void BindDensityBuffer(short* pBuffer, cudaExtent Extent)
{
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<short>();

	HandleCudaError(cudaMalloc3DArray(&gpDensityArray, &ChannelDesc, Extent));

	cudaMemcpy3DParms CopyParams = {0};

	CopyParams.srcPtr	= make_cudaPitchedPtr(pBuffer, Extent.width * sizeof(short), Extent.width, Extent.height);
	CopyParams.dstArray	= gpDensityArray;
	CopyParams.extent	= Extent;
	CopyParams.kind		= cudaMemcpyHostToDevice;
	
	HandleCudaError(cudaMemcpy3D(&CopyParams));

	gTexDensity.normalized		= true;
	gTexDensity.filterMode		= cudaFilterModeLinear;      
	gTexDensity.addressMode[0]	= cudaAddressModeClamp;  
	gTexDensity.addressMode[1]	= cudaAddressModeClamp;
  	gTexDensity.addressMode[2]	= cudaAddressModeClamp;

	HandleCudaError(cudaBindTextureToArray(gTexDensity, gpDensityArray, ChannelDesc));
}

void BindGradientMagnitudeBuffer(short* pBuffer, cudaExtent Extent)
{
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<short>();
	HandleCudaError(cudaMalloc3DArray(&gpGradientMagnitudeArray, &ChannelDesc, Extent));

	cudaMemcpy3DParms CopyParams = {0};

	CopyParams.srcPtr	= make_cudaPitchedPtr(pBuffer, Extent.width * sizeof(short), Extent.width, Extent.height);
	CopyParams.dstArray	= gpGradientMagnitudeArray;
	CopyParams.extent	= Extent;
	CopyParams.kind		= cudaMemcpyHostToDevice;
	
	HandleCudaError(cudaMemcpy3D(&CopyParams));

	gTexGradientMagnitude.normalized		= true;
	gTexGradientMagnitude.filterMode		= cudaFilterModeLinear;      
	gTexGradientMagnitude.addressMode[0]	= cudaAddressModeClamp;  
	gTexGradientMagnitude.addressMode[1]	= cudaAddressModeClamp;
  	gTexGradientMagnitude.addressMode[2]	= cudaAddressModeClamp;

	HandleCudaError(cudaBindTextureToArray(gTexGradientMagnitude, gpGradientMagnitudeArray, ChannelDesc));
}

void UnbindDensityBuffer(void)
{
	HandleCudaError(cudaFreeArray(gpDensityArray));
	gpDensityArray = NULL;
	HandleCudaError(cudaUnbindTexture(gTexDensity));
}

void UnbindGradientMagnitudeBuffer(void)
{
	HandleCudaError(cudaFreeArray(gpGradientMagnitudeArray));
	gpGradientMagnitudeArray = NULL;
	HandleCudaError(cudaUnbindTexture(gTexGradientMagnitude));
}

void BindRenderCanvasView(const CResolution2D& Resolution)
{
	gRenderCanvasView.Resize(Resolution);

	cudaChannelFormatDesc Des = cudaCreateChannelDesc<uchar4>();

	HandleCudaError(cudaBindTexture2D(0, gTexRunningEstimateRgba, gRenderCanvasView.m_EstimateRgbaLdr.GetPtr(0, 0), Des, gRenderCanvasView.m_Resolution.GetResX(), gRenderCanvasView.m_Resolution.GetResY(), gRenderCanvasView.m_EstimateRgbaLdr.GetPitch()));
}

void ResetRenderCanvasView(void)
{
	gRenderCanvasView.Reset();
}

void FreeRenderCanvasView(void)
{
	gRenderCanvasView.Free();
}

unsigned char* GetDisplayEstimate(void)
{
	return (unsigned char*)gRenderCanvasView.m_DisplayEstimateRgbLdr.GetPtr(0, 0);
}

void BindTransferFunctionOpacity(CTransferFunction& TransferFunctionOpacity)
{
	gTexOpacity.normalized		= true;
	gTexOpacity.filterMode		= cudaFilterModeLinear;
	gTexOpacity.addressMode[0]	= cudaAddressModeClamp;

	float Opacity[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
		Opacity[i] = TransferFunctionOpacity.F((float)i * INV_TF_NO_SAMPLES).r;
	
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float>();

	if (gpOpacityArray == NULL)
		HandleCudaError(cudaMallocArray(&gpOpacityArray, &ChannelDesc, TF_NO_SAMPLES, 1));

	HandleCudaError(cudaMemcpyToArray(gpOpacityArray, 0, 0, Opacity, TF_NO_SAMPLES * sizeof(float), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexOpacity, gpOpacityArray, ChannelDesc));
}

void UnbindTransferFunctionOpacity(void)
{
	HandleCudaError(cudaFreeArray(gpOpacityArray));
	gpOpacityArray = NULL;
	HandleCudaError(cudaUnbindTexture(gTexOpacity));
}

void BindTransferFunctionDiffuse(CTransferFunction& TransferFunctionDiffuse)
{
	gTexDiffuse.normalized		= true;
	gTexDiffuse.filterMode		= cudaFilterModeLinear;
	gTexDiffuse.addressMode[0]	= cudaAddressModeClamp;

	float4 Diffuse[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
	{
		Diffuse[i].x = TransferFunctionDiffuse.F((float)i * INV_TF_NO_SAMPLES).r;
		Diffuse[i].y = TransferFunctionDiffuse.F((float)i * INV_TF_NO_SAMPLES).g;
		Diffuse[i].z = TransferFunctionDiffuse.F((float)i * INV_TF_NO_SAMPLES).b;
	}

	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float4>();
	
	if (gpDiffuseArray == NULL)
		HandleCudaError(cudaMallocArray(&gpDiffuseArray, &ChannelDesc, TF_NO_SAMPLES, 1));

	HandleCudaError(cudaMemcpyToArray(gpDiffuseArray, 0, 0, Diffuse, TF_NO_SAMPLES * sizeof(float4), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexDiffuse, gpDiffuseArray, ChannelDesc));
}

void UnbindTransferFunctionDiffuse(void)
{
	HandleCudaError(cudaFreeArray(gpDiffuseArray));
	gpDiffuseArray = NULL;
	HandleCudaError(cudaUnbindTexture(gTexDiffuse));
}

void BindTransferFunctionSpecular(CTransferFunction& TransferFunctionSpecular)
{
	gTexSpecular.normalized		= true;
	gTexSpecular.filterMode		= cudaFilterModeLinear;
	gTexSpecular.addressMode[0]	= cudaAddressModeClamp;

	float4 Specular[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
	{
		Specular[i].x = TransferFunctionSpecular.F((float)i * INV_TF_NO_SAMPLES).r;
		Specular[i].y = TransferFunctionSpecular.F((float)i * INV_TF_NO_SAMPLES).g;
		Specular[i].z = TransferFunctionSpecular.F((float)i * INV_TF_NO_SAMPLES).b;
	}

	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float4>();
	
	if (gpSpecularArray == NULL)
		HandleCudaError(cudaMallocArray(&gpSpecularArray, &ChannelDesc, TF_NO_SAMPLES, 1));

	HandleCudaError(cudaMemcpyToArray(gpSpecularArray, 0, 0, Specular, TF_NO_SAMPLES * sizeof(float4), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexSpecular, gpSpecularArray, ChannelDesc));
}

void UnbindTransferFunctionSpecular(void)
{
	HandleCudaError(cudaFreeArray(gpSpecularArray));
	gpSpecularArray = NULL;
	HandleCudaError(cudaUnbindTexture(gTexSpecular));
}

void BindTransferFunctionRoughness(CTransferFunction& TransferFunctionRoughness)
{
	gTexRoughness.normalized		= true;
	gTexRoughness.filterMode		= cudaFilterModeLinear;
	gTexRoughness.addressMode[0]	= cudaAddressModeClamp;

	float Roughness[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
		Roughness[i] = TransferFunctionRoughness.F((float)i * INV_TF_NO_SAMPLES).r;
	
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float>();

	if (gpRoughnessArray == NULL)
		HandleCudaError(cudaMallocArray(&gpRoughnessArray, &ChannelDesc, TF_NO_SAMPLES, 1));

	HandleCudaError(cudaMemcpyToArray(gpRoughnessArray, 0, 0, Roughness, TF_NO_SAMPLES * sizeof(float),  cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexRoughness, gpRoughnessArray, ChannelDesc));
}

void UnbindTransferFunctionRoughness(void)
{
	HandleCudaError(cudaFreeArray(gpRoughnessArray));
	gpRoughnessArray = NULL;
	HandleCudaError(cudaUnbindTexture(gTexRoughness));
}

void BindTransferFunctionEmission(CTransferFunction& TransferFunctionEmission)
{
	gTexEmission.normalized		= true;
	gTexEmission.filterMode		= cudaFilterModeLinear;
	gTexEmission.addressMode[0]	= cudaAddressModeClamp;

	float4 Emission[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
	{
		Emission[i].x = TransferFunctionEmission.F((float)i * INV_TF_NO_SAMPLES).r;
		Emission[i].y = TransferFunctionEmission.F((float)i * INV_TF_NO_SAMPLES).g;
		Emission[i].z = TransferFunctionEmission.F((float)i * INV_TF_NO_SAMPLES).b;
	}

	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float4>();
	
	if (gpEmissionArray == NULL)
		HandleCudaError(cudaMallocArray(&gpEmissionArray, &ChannelDesc, TF_NO_SAMPLES, 1));

	HandleCudaError(cudaMemcpyToArray(gpEmissionArray, 0, 0, Emission, TF_NO_SAMPLES * sizeof(float4),  cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexEmission, gpEmissionArray, ChannelDesc));
}

void UnbindTransferFunctionEmission(void)
{
	HandleCudaError(cudaFreeArray(gpEmissionArray));
	gpEmissionArray = NULL;
	HandleCudaError(cudaUnbindTexture(gTexEmission));
}

void BindConstants(CScene* pScene)
{
	const float3 AaBbMin = make_float3(pScene->m_BoundingBox.GetMinP().x, pScene->m_BoundingBox.GetMinP().y, pScene->m_BoundingBox.GetMinP().z);
	const float3 AaBbMax = make_float3(pScene->m_BoundingBox.GetMaxP().x, pScene->m_BoundingBox.GetMaxP().y, pScene->m_BoundingBox.GetMaxP().z);

	HandleCudaError(cudaMemcpyToSymbol("gAaBbMin", &AaBbMin, sizeof(float3)));
	HandleCudaError(cudaMemcpyToSymbol("gAaBbMax", &AaBbMax, sizeof(float3)));

	const float3 InvAaBbMin = make_float3(pScene->m_BoundingBox.GetInvMinP().x, pScene->m_BoundingBox.GetInvMinP().y, pScene->m_BoundingBox.GetInvMinP().z);
	const float3 InvAaBbMax = make_float3(pScene->m_BoundingBox.GetInvMaxP().x, pScene->m_BoundingBox.GetInvMaxP().y, pScene->m_BoundingBox.GetInvMaxP().z);

	HandleCudaError(cudaMemcpyToSymbol("gInvAaBbMin", &InvAaBbMin, sizeof(float3)));
	HandleCudaError(cudaMemcpyToSymbol("gInvAaBbMax", &InvAaBbMax, sizeof(float3)));

	const float IntensityMin		= pScene->m_IntensityRange.GetMin();
	const float IntensityMax		= pScene->m_IntensityRange.GetMax();
	const float IntensityRange		= pScene->m_IntensityRange.GetRange();
	const float IntensityInvRange	= 1.0f / IntensityRange;

	HandleCudaError(cudaMemcpyToSymbol("gIntensityMin", &IntensityMin, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gIntensityMax", &IntensityMax, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gIntensityRange", &IntensityRange, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gIntensityInvRange", &IntensityInvRange, sizeof(float)));

	const float StepSize		= pScene->m_StepSizeFactor * pScene->m_GradientDelta;
	const float StepSizeShadow	= pScene->m_StepSizeFactorShadow * pScene->m_GradientDelta;

	HandleCudaError(cudaMemcpyToSymbol("gStepSize", &StepSize, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gStepSizeShadow", &StepSizeShadow, sizeof(float)));

	const float DensityScale = pScene->m_DensityScale;

	HandleCudaError(cudaMemcpyToSymbol("gDensityScale", &DensityScale, sizeof(float)));
	
	const float GradientDelta		= pScene->m_GradientDelta;
	const float InvGradientDelta	= 1.0f / (2.0f * GradientDelta);
	const Vec3f GradientDeltaX(GradientDelta, 0.0f, 0.0f);
	const Vec3f GradientDeltaY(0.0f, GradientDelta, 0.0f);
	const Vec3f GradientDeltaZ(0.0f, 0.0f, GradientDelta);
	
	HandleCudaError(cudaMemcpyToSymbol("gGradientDelta", &GradientDelta, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gInvGradientDelta", &InvGradientDelta, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gGradientDeltaX", &GradientDeltaX, sizeof(Vec3f)));
	HandleCudaError(cudaMemcpyToSymbol("gGradientDeltaY", &GradientDeltaY, sizeof(Vec3f)));
	HandleCudaError(cudaMemcpyToSymbol("gGradientDeltaZ", &GradientDeltaZ, sizeof(Vec3f)));
	
	const int FilmWidth		= pScene->m_Camera.m_Film.GetWidth();
	const int Filmheight	= pScene->m_Camera.m_Film.GetHeight();
	const int FilmNoPixels	= pScene->m_Camera.m_Film.m_Resolution.GetNoElements();

	HandleCudaError(cudaMemcpyToSymbol("gFilmWidth", &FilmWidth, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol("gFilmHeight", &Filmheight, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol("gFilmNoPixels", &FilmNoPixels, sizeof(int)));

	const int FilterWidth = 2;

	HandleCudaError(cudaMemcpyToSymbol("gFilterWidth", &FilterWidth, sizeof(int)));

	const float FilterWeights[10] = { 0.11411459588254977f, 0.08176668094332218f, 0.03008028089187349f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

	HandleCudaError(cudaMemcpyToSymbol("gFilterWeights", &FilterWeights, 10 * sizeof(float)));

	const float Gamma		= pScene->m_Camera.m_Film.m_Gamma;
	const float InvGamma	= 1.0f / Gamma;
	const float Exposure	= pScene->m_Camera.m_Film.m_Exposure;
	const float InvExposure	= 1.0f / Exposure;

	HandleCudaError(cudaMemcpyToSymbol("gExposure", &Exposure, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gInvExposure", &InvExposure, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gGamma", &Gamma, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gInvGamma", &InvGamma, sizeof(float)));

	HandleCudaError(cudaMemcpyToSymbol("gDenoiseEnabled", &pScene->m_DenoiseParams.m_Enabled, sizeof(bool)));
	HandleCudaError(cudaMemcpyToSymbol("gDenoiseWindowRadius", &pScene->m_DenoiseParams.m_WindowRadius, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gDenoiseInvWindowArea", &pScene->m_DenoiseParams.m_InvWindowArea, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gDenoiseNoise", &pScene->m_DenoiseParams.m_Noise, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gDenoiseWeightThreshold", &pScene->m_DenoiseParams.m_WeightThreshold, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gDenoiseLerpThreshold", &pScene->m_DenoiseParams.m_LerpThreshold, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gDenoiseLerpC", &pScene->m_DenoiseParams.m_LerpC, sizeof(float)));

	const float NoIterations	= pScene->GetNoIterations();
	const float InvNoIterations = 1.0f / __max(1.0f, NoIterations);

	HandleCudaError(cudaMemcpyToSymbol("gNoIterations", &NoIterations, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gInvNoIterations", &InvNoIterations, sizeof(float)));
}

void Render(const int& Type, CScene& Scene, CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage)
{
	CScene* pDevScene = NULL;

	HandleCudaError(cudaMalloc(&pDevScene, sizeof(CScene)));
	HandleCudaError(cudaMemcpy(pDevScene, &Scene, sizeof(CScene), cudaMemcpyHostToDevice));

	if (Scene.m_Camera.m_Focus.m_Type == 0)
		Scene.m_Camera.m_Focus.m_FocalDistance = NearestIntersection(pDevScene);

	HandleCudaError(cudaMemcpy(pDevScene, &Scene, sizeof(CScene), cudaMemcpyHostToDevice));

	CCudaView* pDevView = NULL;

	HandleCudaError(cudaMalloc(&pDevView, sizeof(CCudaView)));
	HandleCudaError(cudaMemcpy(pDevView, &gRenderCanvasView, sizeof(CCudaView), cudaMemcpyHostToDevice));

	
	CCudaTimer TmrRender;
	
	switch (Type)
	{
		case 0:
		{
			SingleScattering(&Scene, pDevScene, pDevView);
			break;
		}

		case 1:
		{
//			MultipleScattering(&Scene, pDevScene);
			break;
		}
	}

	RenderImage.AddDuration(TmrRender.ElapsedTime());
	
 	CCudaTimer TmrBlur;
	BlurImageXyz(&Scene, pDevScene, pDevView);
	BlurImage.AddDuration(TmrBlur.ElapsedTime());

	CCudaTimer TmrPostProcess;
	Estimate(&Scene, pDevScene, pDevView);
	PostProcessImage.AddDuration(TmrPostProcess.ElapsedTime());

	ToneMap(&Scene, pDevScene, pDevView);

	CCudaTimer TmrDenoise;
	Denoise(&Scene, pDevScene, pDevView);
	DenoiseImage.AddDuration(TmrDenoise.ElapsedTime());
	

	HandleCudaError(cudaFree(pDevScene));
	HandleCudaError(cudaFree(pDevView));
}