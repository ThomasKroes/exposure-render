
#include "Core.cuh"

texture<short, 3, cudaReadModeNormalizedFloat>		gTexDensity;
texture<short, 3, cudaReadModeNormalizedFloat>		gTexGradientMagnitude;
texture<float, 3, cudaReadModeElementType>			gTexExtinction;
texture<uchar4, 2, cudaReadModeNormalizedFloat>		gTexEstimateRgbLdr;
texture<float, 1, cudaReadModeElementType>			gTexOpacity;
texture<float4, 1, cudaReadModeElementType>			gTexDiffuse;
texture<float4, 1, cudaReadModeElementType>			gTexSpecular;
texture<float, 1, cudaReadModeElementType>			gTexRoughness;
texture<float4, 1, cudaReadModeElementType>			gTexEmission;

cudaArray* gpDensityArray				= NULL;
cudaArray* gpGradientMagnitudeArray		= NULL;
cudaArray* gpOpacityArray				= NULL;
cudaArray* gpDiffuseArray				= NULL;
cudaArray* gpSpecularArray				= NULL;
cudaArray* gpRoughnessArray				= NULL;
cudaArray* gpEmissionArray				= NULL;

__constant__ float3		gAaBbMin;
__constant__ float3		gAaBbMax;
__constant__ float3		gInvAaBbMin;
__constant__ float3		gInvAaBbMax;
__constant__ float		gIntensityMin;
__constant__ float		gIntensityMax;
__constant__ float		gIntensityRange;
__constant__ float		gIntensityInvRange;
__constant__ float		gStepSize;
__constant__ float		gStepSizeShadow;
__constant__ float		gDensityScale;
__constant__ float		gGradientDelta;
__constant__ float		gInvGradientDelta;
__constant__ int		gFilmWidth;
__constant__ int		gFilmHeight;
__constant__ int		gFilmNoPixels;
__constant__ int		gFilterWidth;
__constant__ float		gFilterWeights[3];
__constant__ float		gExposure;
__constant__ float		gInvExposure;
__constant__ float		gGamma;
__constant__ float		gInvGamma;
__constant__ float		gDenoiseEnabled;
__constant__ float		gDenoiseWindowRadius;
__constant__ float		gDenoiseInvWindowArea;
__constant__ float		gDenoiseNoise;
__constant__ float		gDenoiseWeightThreshold;
__constant__ float		gDenoiseLerpThreshold;
__constant__ float		gDenoiseLerpC;

#define TF_NO_SAMPLES		256
#define INV_TF_NO_SAMPLES	0.00390625f

#include "Blur.cuh"
#include "Denoise.cuh"
#include "Estimate.cuh"
#include "Utilities.cuh"
#include "SingleScattering.cuh"
#include "MultipleScattering.cuh"
#include "Variance.cuh"
#include "NearestIntersection.cuh"

#include "CudaUtilities.h"

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

void BindEstimateRgbLdr(CColorRgbaLdr* pBuffer, int Width, int Height)
{
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<uchar4>();

	gTexEstimateRgbLdr.filterMode = cudaFilterModeLinear;     

	HandleCudaError(cudaBindTexture2D(0, gTexEstimateRgbLdr, (void*)pBuffer, ChannelDesc, Width, Height, Width * sizeof(uchar4)));
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
	const float InvGradientDelta	= 1.0f / GradientDelta;

	HandleCudaError(cudaMemcpyToSymbol("gGradientDelta", &GradientDelta, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gInvGradientDelta", &InvGradientDelta, sizeof(float)));
	
	const int FilmWidth		= pScene->m_Camera.m_Film.GetWidth();
	const int Filmheight	= pScene->m_Camera.m_Film.GetHeight();
	const int FilmNoPixels	= pScene->m_Camera.m_Film.m_Resolution.GetNoElements();

	HandleCudaError(cudaMemcpyToSymbol("gFilmWidth", &FilmWidth, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol("gFilmHeight", &Filmheight, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol("gFilmNoPixels", &FilmNoPixels, sizeof(int)));

	const int FilterWidth = 2;

	HandleCudaError(cudaMemcpyToSymbol("gFilterWidth", &FilterWidth, sizeof(int)));

	const float FilterWeights[3] = { 0.11411459588254977f, 0.08176668094332218f, 0.03008028089187349f };

	HandleCudaError(cudaMemcpyToSymbol("gFilterWeights", &FilterWeights, 3 * sizeof(float)));

	const float Gamma		= pScene->m_Camera.m_Film.m_Gamma;
	const float InvGamma	= 1.0f / Gamma;
	const float Exposure	= pScene->m_Camera.m_Film.m_Exposure;
	const float InvExposure	= 1.0f / Exposure;

	HandleCudaError(cudaMemcpyToSymbol("gExposure", &Exposure, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gInvExposure", &InvExposure, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gGamma", &Gamma, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gInvGamma", &InvGamma, sizeof(float)));

	HandleCudaError(cudaMemcpyToSymbol("gDenoiseEnabled", &pScene->m_DenoiseParams.m_Enabled, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gDenoiseWindowRadius", &pScene->m_DenoiseParams.m_WindowRadius, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gDenoiseInvWindowArea", &pScene->m_DenoiseParams.m_InvWindowArea, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gDenoiseNoise", &pScene->m_DenoiseParams.m_Noise, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gDenoiseWeightThreshold", &pScene->m_DenoiseParams.m_WeightThreshold, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gDenoiseLerpThreshold", &pScene->m_DenoiseParams.m_LerpThreshold, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol("gDenoiseLerpC", &pScene->m_DenoiseParams.m_LerpC, sizeof(float)));
}

void Render(const int& Type, CScene& Scene, CCudaFrameBuffers& CudaFrameBuffers, int N, CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage)
{
	CScene* pDevScene = NULL;

	HandleCudaError(cudaMalloc(&pDevScene, sizeof(CScene)));
	HandleCudaError(cudaMemcpy(pDevScene, &Scene, sizeof(CScene), cudaMemcpyHostToDevice));

	if (Scene.m_Camera.m_Focus.m_Type == 0)
		Scene.m_Camera.m_Focus.m_FocalDistance = NearestIntersection(pDevScene);

	HandleCudaError(cudaMemcpy(pDevScene, &Scene, sizeof(CScene), cudaMemcpyHostToDevice));

	CCudaTimer TmrRender;
	
	switch (Type)
	{
		case 0:
		{
			SingleScattering(&Scene, pDevScene, CudaFrameBuffers.m_pDevSeeds, CudaFrameBuffers.m_pDevEstFrameXyz);
			HandleCudaError(cudaGetLastError());
			break;
		}

		case 1:
		{
			MultipleScattering(&Scene, pDevScene, CudaFrameBuffers.m_pDevSeeds, CudaFrameBuffers.m_pDevEstFrameXyz);
			HandleCudaError(cudaGetLastError());
			break;
		}
	}

	RenderImage.AddDuration(TmrRender.ElapsedTime());
	
 	CCudaTimer TmrBlur;
	BlurImageXyz(&Scene, pDevScene, CudaFrameBuffers.m_pDevEstFrameXyz, CudaFrameBuffers.m_pDevEstFrameBlurXyz);
	HandleCudaError(cudaGetLastError());
	BlurImage.AddDuration(TmrBlur.ElapsedTime());

	CCudaTimer TmrPostProcess;
	Estimate(&Scene, pDevScene, CudaFrameBuffers.m_pDevEstFrameXyz, CudaFrameBuffers.m_pDevAccEstXyz, CudaFrameBuffers.m_pDevEstXyz, CudaFrameBuffers.m_pDevEstRgbaLdr, N);
	HandleCudaError(cudaGetLastError());
	PostProcessImage.AddDuration(TmrPostProcess.ElapsedTime());

	CCudaTimer TmrDenoise;
	Denoise(&Scene, pDevScene, CudaFrameBuffers.m_pDevEstRgbaLdr, CudaFrameBuffers.m_pDevRgbLdrDisp);
	HandleCudaError(cudaGetLastError());
	DenoiseImage.AddDuration(TmrDenoise.ElapsedTime());

	HandleCudaError(cudaFree(pDevScene));
}