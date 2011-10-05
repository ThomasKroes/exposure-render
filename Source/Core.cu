
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

#define TF_NO_SAMPLES		256
#define INV_TF_NO_SAMPLES	0.00390625f

#include "Blur.cuh"
#include "Denoise.cuh"
#include "Estimate.cuh"
#include "Utilities.cuh"
#include "SingleScattering.cuh"
#include "MultipleScattering.cuh"
#include "Variance.cuh"

#include "CudaUtilities.h"

void BindDensityBuffer(short* pBuffer, cudaExtent Extent)
{
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<short>();
	cudaMalloc3DArray(&gpDensityArray, &ChannelDesc, Extent);

	cudaMemcpy3DParms CopyParams = {0};

	CopyParams.srcPtr	= make_cudaPitchedPtr(pBuffer, Extent.width * sizeof(short), Extent.width, Extent.height);
	CopyParams.dstArray	= gpDensityArray;
	CopyParams.extent	= Extent;
	CopyParams.kind		= cudaMemcpyHostToDevice;
	
	cudaMemcpy3D(&CopyParams);

	gTexDensity.normalized		= true;
	gTexDensity.filterMode		= cudaFilterModeLinear;      
	gTexDensity.addressMode[0]	= cudaAddressModeClamp;  
	gTexDensity.addressMode[1]	= cudaAddressModeClamp;
  	gTexDensity.addressMode[2]	= cudaAddressModeClamp;

	cudaBindTextureToArray(gTexDensity, gpDensityArray, ChannelDesc);
}

void BindGradientMagnitudeBuffer(short* pBuffer, cudaExtent Extent)
{
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<short>();
	cudaMalloc3DArray(&gpGradientMagnitudeArray, &ChannelDesc, Extent);

	cudaMemcpy3DParms CopyParams = {0};

	CopyParams.srcPtr	= make_cudaPitchedPtr(pBuffer, Extent.width * sizeof(short), Extent.width, Extent.height);
	CopyParams.dstArray	= gpGradientMagnitudeArray;
	CopyParams.extent	= Extent;
	CopyParams.kind		= cudaMemcpyHostToDevice;
	
	cudaMemcpy3D(&CopyParams);

	gTexGradientMagnitude.normalized		= true;
	gTexGradientMagnitude.filterMode		= cudaFilterModeLinear;      
	gTexGradientMagnitude.addressMode[0]	= cudaAddressModeClamp;  
	gTexGradientMagnitude.addressMode[1]	= cudaAddressModeClamp;
  	gTexGradientMagnitude.addressMode[2]	= cudaAddressModeClamp;

	cudaBindTextureToArray(gTexGradientMagnitude, gpGradientMagnitudeArray, ChannelDesc);
}

void UnbindDensityBuffer(void)
{
	cudaFreeArray(gpDensityArray);
	gpDensityArray = NULL;
	cudaUnbindTexture(gTexDensity);
}

void UnbindGradientMagnitudeBuffer(void)
{
	cudaFreeArray(gpGradientMagnitudeArray);
	gpGradientMagnitudeArray = NULL;
	cudaUnbindTexture(gTexGradientMagnitude);
}

void BindEstimateRgbLdr(CColorRgbaLdr* pBuffer, int Width, int Height)
{
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<uchar4>();

	cudaBindTexture2D(0, gTexEstimateRgbLdr, (void*)pBuffer, ChannelDesc, Width, Height, Width * sizeof(uchar4));
}

void BindTransferFunctions(CTransferFunctions& TransferFunctions)
{
	cudaChannelFormatDesc ChannelDesc;

	gTexOpacity.normalized			= true;
	gTexDiffuse.normalized			= true;
	gTexSpecular.normalized			= true;
	gTexRoughness.normalized		= true;
	gTexEmission.normalized			= true;

	gTexOpacity.filterMode			= cudaFilterModeLinear;
	gTexDiffuse.filterMode			= cudaFilterModeLinear;
	gTexSpecular.filterMode			= cudaFilterModeLinear;
	gTexRoughness.filterMode		= cudaFilterModeLinear;
	gTexEmission.filterMode			= cudaFilterModeLinear;

	gTexOpacity.addressMode[0]		= cudaAddressModeClamp;
	gTexDiffuse.addressMode[0]		= cudaAddressModeClamp;
	gTexSpecular.addressMode[0]		= cudaAddressModeClamp;
	gTexRoughness.addressMode[0]	= cudaAddressModeClamp;
	gTexEmission.addressMode[0]		= cudaAddressModeClamp;

	// Opacity
	float Opacity[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
		Opacity[i] = TransferFunctions.m_Opacity.F((float)i * INV_TF_NO_SAMPLES).r;
	
	ChannelDesc = cudaCreateChannelDesc<float>();

	if (gpOpacityArray == NULL)
		cudaMallocArray(&gpOpacityArray, &ChannelDesc, TF_NO_SAMPLES, 1);

	cudaMemcpyToArray(gpOpacityArray, 0, 0, Opacity, TF_NO_SAMPLES * sizeof(float),  cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexOpacity, gpOpacityArray, ChannelDesc);
//	float4* pDevOpacity = NULL;
//	cudaMalloc(&pDevOpacity, 256 * sizeof(float4));
//	cudaBindTexture(0, gTexOpacity, pDevOpacity, ChannelDesc, 256 * sizeof(float4));

	// Diffuse
	float4 Diffuse[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
	{
		Diffuse[i].x = TransferFunctions.m_Diffuse.F((float)i * INV_TF_NO_SAMPLES).r;
		Diffuse[i].y = TransferFunctions.m_Diffuse.F((float)i * INV_TF_NO_SAMPLES).g;
		Diffuse[i].z = TransferFunctions.m_Diffuse.F((float)i * INV_TF_NO_SAMPLES).b;
	}

	ChannelDesc = cudaCreateChannelDesc<float4>();
	
	if (gpDiffuseArray == NULL)
		cudaMallocArray(&gpDiffuseArray, &ChannelDesc, TF_NO_SAMPLES, 1);

	cudaMemcpyToArray(gpDiffuseArray, 0, 0, Diffuse, TF_NO_SAMPLES * sizeof(float4),  cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexDiffuse, gpDiffuseArray, ChannelDesc);

	// Specular
	float4 Specular[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
	{
		Specular[i].x = TransferFunctions.m_Specular.F((float)i * INV_TF_NO_SAMPLES).r;
		Specular[i].y = TransferFunctions.m_Specular.F((float)i * INV_TF_NO_SAMPLES).g;
		Specular[i].z = TransferFunctions.m_Specular.F((float)i * INV_TF_NO_SAMPLES).b;
	}

	ChannelDesc = cudaCreateChannelDesc<float4>();
	
	if (gpSpecularArray == NULL)
		cudaMallocArray(&gpSpecularArray, &ChannelDesc, TF_NO_SAMPLES, 1);

	cudaMemcpyToArray(gpSpecularArray, 0, 0, Specular, TF_NO_SAMPLES * sizeof(float4),  cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexSpecular, gpSpecularArray, ChannelDesc);

	// Roughness
	float Roughness[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
		Roughness[i] = TransferFunctions.m_Roughness.F((float)i * INV_TF_NO_SAMPLES).r;
	
	if (gpRoughnessArray == NULL)
		cudaMallocArray(&gpRoughnessArray, &ChannelDesc, TF_NO_SAMPLES, 1);

	cudaMemcpyToArray(gpRoughnessArray, 0, 0, Roughness, TF_NO_SAMPLES * sizeof(float),  cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexRoughness, gpRoughnessArray, ChannelDesc);

	// Emission
	float4 Emission[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
	{
		Emission[i].x = TransferFunctions.m_Emission.F((float)i * INV_TF_NO_SAMPLES).r;
		Emission[i].y = TransferFunctions.m_Emission.F((float)i * INV_TF_NO_SAMPLES).g;
		Emission[i].z = TransferFunctions.m_Emission.F((float)i * INV_TF_NO_SAMPLES).b;
	}

	ChannelDesc = cudaCreateChannelDesc<float4>();
	
	if (gpEmissionArray == NULL)
		cudaMallocArray(&gpEmissionArray, &ChannelDesc, TF_NO_SAMPLES, 1);

	cudaMemcpyToArray(gpEmissionArray, 0, 0, Emission, TF_NO_SAMPLES * sizeof(float4),  cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexEmission, gpEmissionArray, ChannelDesc);
}

void Render(const int& Type, CScene* pScene, CScene* pDevScene, CCudaFrameBuffers& CudaFrameBuffers, int N, CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage)
{
	CCudaTimer TmrRender;
	
	switch (Type)
	{
		case 0:
		{
			SingleScattering(pScene, pDevScene, CudaFrameBuffers.m_pDevSeeds, CudaFrameBuffers.m_pDevEstFrameXyz);
			HandleCudaError(cudaGetLastError());
			break;
		}

		case 1:
		{
			MultipleScattering(pScene, pDevScene, CudaFrameBuffers.m_pDevSeeds, CudaFrameBuffers.m_pDevEstFrameXyz);
			HandleCudaError(cudaGetLastError());
			break;
		}
	}

	RenderImage.AddDuration(TmrRender.ElapsedTime());
	
 	CCudaTimer TmrBlur;
	BlurImageXyz(CudaFrameBuffers.m_pDevEstFrameXyz, CudaFrameBuffers.m_pDevEstFrameBlurXyz, CResolution2D(pScene->m_Camera.m_Film.m_Resolution.GetResX(), pScene->m_Camera.m_Film.m_Resolution.GetResY()), 5.0f);
	HandleCudaError(cudaGetLastError());
	BlurImage.AddDuration(TmrBlur.ElapsedTime());

	CCudaTimer TmrPostProcess;
	Estimate(pScene, pDevScene, CudaFrameBuffers.m_pDevEstFrameXyz, CudaFrameBuffers.m_pDevAccEstXyz, CudaFrameBuffers.m_pDevEstXyz, CudaFrameBuffers.m_pDevEstRgbaLdr, N);
	HandleCudaError(cudaGetLastError());
	PostProcessImage.AddDuration(TmrPostProcess.ElapsedTime());

	CCudaTimer TmrDenoise;
	Denoise(pScene, pDevScene, CudaFrameBuffers.m_pDevEstRgbaLdr, CudaFrameBuffers.m_pDevRgbLdrDisp);
	HandleCudaError(cudaGetLastError());
	DenoiseImage.AddDuration(TmrDenoise.ElapsedTime());
}