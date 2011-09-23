
#include "Core.cuh"

texture<float, 3, cudaReadModeElementType>				gTexDensity;
texture<short, 3, cudaReadModeNormalizedFloat>			gTexExtinction;
texture<float, 3, cudaReadModeElementType>				gTexGradientMagnitude;
texture<unsigned char, 3, cudaReadModeNormalizedFloat>	gTexEstimateRgbLdr;

#include "Blur.cuh"
#include "Denoise.cuh"
#include "ComputeEstimate.cuh"
#include "Utilities.cuh"
#include "SingleScattering.cuh"
#include "MultipleScattering.cuh"

#include "CudaUtilities.h"

void BindDensityVolume(float* densityBuffer, cudaExtent Extent)
{
	cudaArray* pArray = NULL;

	// create 3D array
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&pArray, &ChannelDesc, Extent);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams	= {0};
	copyParams.srcPtr				= make_cudaPitchedPtr(densityBuffer, Extent.width * sizeof(float), Extent.width, Extent.height);
	copyParams.dstArray				= pArray;
	copyParams.extent				= Extent;
	copyParams.kind					= cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	// Set texture parameters
	gTexDensity.normalized			= true;
	gTexDensity.filterMode			= cudaFilterModeLinear;      
	gTexDensity.addressMode[0]		= cudaAddressModeClamp;  
	gTexDensity.addressMode[1]		= cudaAddressModeClamp;
 	gTexDensity.addressMode[2]		= cudaAddressModeClamp;

	// Bind array to 3D texture
	cudaBindTextureToArray(gTexDensity, pArray, ChannelDesc);
}

void BindExtinctionVolume(float* extinction, cudaExtent Extent)
{
	cudaArray* volArray;
	cudaExtent volExtent = make_cudaExtent(Extent.width, Extent.height, Extent.depth);
	cudaChannelFormatDesc volChannelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&volArray, &volChannelDesc, volExtent);
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = make_cudaPitchedPtr((void*)extinction, Extent.width * sizeof(float), Extent.width, Extent.height);
	copyParams.dstArray = volArray;
	copyParams.extent = Extent;
	copyParams.kind = cudaMemcpyHostToDevice;
	CUDA_SAFE_CALL( cudaMemcpy3D(&copyParams));

	gTexExtinction.normalized = true;
	gTexExtinction.filterMode = cudaFilterModePoint;
	gTexExtinction.addressMode[0] = cudaAddressModeClamp;
	gTexExtinction.addressMode[1] = cudaAddressModeClamp;

	cudaBindTextureToArray(gTexExtinction, volArray, volChannelDesc);
}

void BindGradientMagnitudeVolume(float* pBuffer, cudaExtent Extent)
{
	cudaArray* pVolumeArray;

	cudaChannelFormatDesc VolumeChannelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&pVolumeArray, &VolumeChannelDesc, Extent);
	
	cudaMemcpy3DParms CopyParams = {0};

	CopyParams.srcPtr	= make_cudaPitchedPtr((void*)pBuffer, Extent.width * sizeof(float), Extent.width, Extent.height);
	CopyParams.dstArray = pVolumeArray;
	CopyParams.extent	= Extent;
	CopyParams.kind		= cudaMemcpyHostToDevice;

	CUDA_SAFE_CALL(cudaMemcpy3D(&CopyParams));

	gTexGradientMagnitude.normalized		= true;
	gTexGradientMagnitude.filterMode		= cudaFilterModePoint;
	gTexGradientMagnitude.addressMode[0]	= cudaAddressModeClamp;
	gTexGradientMagnitude.addressMode[1]	= cudaAddressModeClamp;
	gTexGradientMagnitude.addressMode[2]	= cudaAddressModeClamp;

	cudaBindTextureToArray(gTexGradientMagnitude, pVolumeArray, VolumeChannelDesc);
}

void BindEstimateRgbLdr(unsigned char* pBuffer, int Width, int Height)
{
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<unsigned char>();

	cudaBindTexture2D(0, gTexEstimateRgbLdr, pBuffer, ChannelDesc, Width, Height, Width * sizeof(unsigned char));
}


void Render(const int& Type, CScene* pScene, CScene* pDevScene, unsigned int* pSeeds, CColorXyz* pDevEstFrameXyz, CColorXyz* pDevEstFrameBlurXyz, CColorXyz* pDevAccEstXyz, unsigned char* pDevEstRgbLdr, unsigned char* pDevEstRgbLdrDisp, int N, CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage)
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
	ComputeEstimate(pScene->m_Camera.m_Film.m_Resolution.GetResX(), pScene->m_Camera.m_Film.m_Resolution.GetResY(), pDevEstFrameXyz, pDevAccEstXyz, N, pScene->m_Camera.m_Film.m_Exposure, pDevEstRgbLdr);
	HandleCudaError(cudaGetLastError());
	PostProcessImage.AddDuration(TmrPostProcess.ElapsedTime());

	CCudaTimer TmrDenoise;
//	Denoise(pScene, pDevScene, (CColorRgbLdr*)pDevEstRgbLdr, (CColorRgbLdr*)pDevEstRgbLdrDisp);
	HandleCudaError(cudaGetLastError());
	DenoiseImage.AddDuration(TmrDenoise.ElapsedTime());
}