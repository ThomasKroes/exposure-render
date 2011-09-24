
#include "Core.cuh"

texture<float, 3, cudaReadModeElementType>			gTexDensity;
texture<float, 3, cudaReadModeElementType>			gTexExtinction;
texture<float, 3, cudaReadModeElementType>			gTexGradientMagnitude;
texture<uchar4, 2, cudaReadModeNormalizedFloat>		gTexEstimateRgbLdr;

#include "Blur.cuh"
#include "Denoise.cuh"
#include "ComputeEstimate.cuh"
#include "Utilities.cuh"
#include "SingleScattering.cuh"
#include "MultipleScattering.cuh"

#include "CudaUtilities.h"

void BindDensityBuffer(float* pBuffer, cudaExtent Extent)
{
	cudaArray* pArray = NULL;

	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&pArray, &ChannelDesc, Extent);

	cudaMemcpy3DParms CopyParams = {0};

	CopyParams.srcPtr				= make_cudaPitchedPtr(pBuffer, Extent.width * sizeof(float), Extent.width, Extent.height);
	CopyParams.dstArray				= pArray;
	CopyParams.extent				= Extent;
	CopyParams.kind					= cudaMemcpyHostToDevice;
	
	cudaMemcpy3D(&CopyParams);

	// Set texture parameters
	gTexDensity.normalized			= true;
	gTexDensity.filterMode			= cudaFilterModeLinear;      
	gTexDensity.addressMode[0]		= cudaAddressModeClamp;  
	gTexDensity.addressMode[1]		= cudaAddressModeClamp;
//  	gTexDensity.addressMode[2]		= cudaAddressModeClamp;

	// Bind array to 3D texture
	cudaBindTextureToArray(gTexDensity, pArray, ChannelDesc);
}

void BindExtinctionBuffer(float* pBuffer, cudaExtent Extent)
{
	cudaArray* pArray;

	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&pArray, &ChannelDesc, Extent);
	
	cudaMemcpy3DParms CopyParams = {0};

	CopyParams.srcPtr				= make_cudaPitchedPtr((void*)pBuffer, Extent.width * sizeof(float), Extent.width, Extent.height);
	CopyParams.dstArray				= pArray;
	CopyParams.extent				= Extent;
	CopyParams.kind					= cudaMemcpyHostToDevice;

	cudaMemcpy3D(&CopyParams);

	gTexExtinction.normalized		= true;
	gTexExtinction.filterMode		= cudaFilterModePoint;
	gTexExtinction.addressMode[0]	= cudaAddressModeClamp;
	gTexExtinction.addressMode[1]	= cudaAddressModeClamp;
// 	gTexExtinction.addressMode[2]	= cudaAddressModeClamp;

	cudaBindTextureToArray(gTexExtinction, pArray, ChannelDesc);
}

void BindGradientMagnitudeBuffer(float* pBuffer, cudaExtent Extent)
{
	cudaArray* pArray;

	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&pArray, &ChannelDesc, Extent);
	
	cudaMemcpy3DParms CopyParams = {0};

	CopyParams.srcPtr	= make_cudaPitchedPtr((void*)pBuffer, Extent.width * sizeof(float), Extent.width, Extent.height);
	CopyParams.dstArray = pArray;
	CopyParams.extent	= Extent;
	CopyParams.kind		= cudaMemcpyHostToDevice;

	CUDA_SAFE_CALL(cudaMemcpy3D(&CopyParams));

	gTexGradientMagnitude.normalized		= true;
	gTexGradientMagnitude.filterMode		= cudaFilterModePoint;
	gTexGradientMagnitude.addressMode[0]	= cudaAddressModeClamp;
	gTexGradientMagnitude.addressMode[1]	= cudaAddressModeClamp;
// 	gTexGradientMagnitude.addressMode[2]	= cudaAddressModeClamp;

	cudaBindTextureToArray(gTexGradientMagnitude, pArray, ChannelDesc);
}

void BindEstimateRgbLdr(unsigned char* pBuffer, int Width, int Height)
{
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<uchar4>();

	cudaBindTexture2D(0, gTexEstimateRgbLdr, pBuffer, ChannelDesc, Width, Height, Width * sizeof(uchar4));
}

void Render(const int& Type, CScene* pScene, CScene* pDevScene, unsigned int* pSeeds, CColorXyz* pDevEstFrameXyz, CColorXyz* pDevEstFrameBlurXyz, CColorXyz* pDevAccEstXyz, CColorRgbaLdr* pDevEstRgbaLdr, unsigned char* pDevEstRgbLdrDisp, int N, CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage)
{
	CCudaTimer TmrRender;
	
	switch (Type)
	{
		case 0:
			{
				SingleScattering(pScene, pDevScene, pSeeds, pDevEstFrameXyz);
				HandleCudaError(cudaGetLastError());
				break;
			}

		case 1:
			{
				MultipleScattering(pScene, pDevScene, pSeeds, pDevEstFrameXyz);
				HandleCudaError(cudaGetLastError());
				break;
			}
	}

	RenderImage.AddDuration(TmrRender.ElapsedTime());

 	CCudaTimer TmrBlur;
	BlurImageXyz(pDevEstFrameXyz, pDevEstFrameBlurXyz, CResolution2D(pScene->m_Camera.m_Film.m_Resolution.GetResX(), pScene->m_Camera.m_Film.m_Resolution.GetResY()), 1.3f);
	HandleCudaError(cudaGetLastError());
	BlurImage.AddDuration(TmrBlur.ElapsedTime());

	CCudaTimer TmrPostProcess;
	ComputeEstimate(pScene->m_Camera.m_Film.m_Resolution.GetResX(), pScene->m_Camera.m_Film.m_Resolution.GetResY(), pDevEstFrameXyz, pDevAccEstXyz, N, pScene->m_Camera.m_Film.m_Exposure, pDevEstRgbaLdr);
	HandleCudaError(cudaGetLastError());
	PostProcessImage.AddDuration(TmrPostProcess.ElapsedTime());

	CCudaTimer TmrDenoise;
	Denoise(pScene, pDevScene, pDevEstRgbaLdr, (CColorRgbLdr*)pDevEstRgbLdrDisp);
	HandleCudaError(cudaGetLastError());
	DenoiseImage.AddDuration(TmrDenoise.ElapsedTime());
}