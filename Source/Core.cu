
#include "Core.cuh"

texture<short, 3, cudaReadModeNormalizedFloat >	gTexDensity;
texture<short, 3, cudaReadModeNormalizedFloat >	gTexExtinction;
texture<short, 3, cudaReadModeNormalizedFloat >	gTexGradientMagnitude;

#include "Blur.cuh"
#include "Denoise.cuh"
#include "ComputeEstimate.cuh"
#include "Utilities.cuh"
#include "SingleScattering.cuh"
#include "MultipleScattering.cuh"

void BindDensityVolume(short* densityBuffer, cudaExtent densityBufferSize)
{
	cudaArray* gpDensity = NULL;

	// create 3D array
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<short>();
	cudaMalloc3DArray(&gpDensity, &ChannelDesc, densityBufferSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams	= {0};
	copyParams.srcPtr				= make_cudaPitchedPtr(densityBuffer, densityBufferSize.width * sizeof(short), densityBufferSize.width, densityBufferSize.height);
	copyParams.dstArray				= gpDensity;
	copyParams.extent				= densityBufferSize;
	copyParams.kind					= cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	// Set texture parameters
	gTexDensity.normalized			= true;
	gTexDensity.filterMode			= cudaFilterModeLinear;      
	gTexDensity.addressMode[0]		= cudaAddressModeClamp;  
	gTexDensity.addressMode[1]		= cudaAddressModeClamp;
 	gTexDensity.addressMode[2]		= cudaAddressModeClamp;

	// Bind array to 3D texture
	cudaBindTextureToArray(gTexDensity, gpDensity, ChannelDesc);
}

void BindExtinctionVolume(float* extinction, cudaExtent extinctionSize)
{
	cudaArray* volArray;
	cudaExtent volExtent = make_cudaExtent(extinctionSize.width, extinctionSize.height, extinctionSize.depth);
	cudaChannelFormatDesc volChannelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&volArray, &volChannelDesc, volExtent);
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = make_cudaPitchedPtr((void*)extinction, volExtent.width * sizeof(float), volExtent.width, volExtent.height);
	copyParams.dstArray = volArray;
	copyParams.extent = volExtent;
	copyParams.kind = cudaMemcpyHostToDevice;
	CUDA_SAFE_CALL( cudaMemcpy3D(&copyParams));

	gTexExtinction.normalized = true;
	gTexExtinction.filterMode = cudaFilterModePoint;
	gTexExtinction.addressMode[0] = cudaAddressModeClamp;
	gTexExtinction.addressMode[1] = cudaAddressModeClamp;

	cudaBindTextureToArray( gTexExtinction, volArray, volChannelDesc);
}

void BindGradientMagnitudeVolume(short* pBuffer, cudaExtent VolumeSize)
{
	cudaArray* pVolumeArray;

	cudaChannelFormatDesc VolumeChannelDesc = cudaCreateChannelDesc<short>();
	cudaMalloc3DArray(&pVolumeArray, &VolumeChannelDesc, VolumeSize);
	
	cudaMemcpy3DParms CopyParams = {0};

	CopyParams.srcPtr	= make_cudaPitchedPtr((void*)pBuffer, VolumeSize.width * sizeof(short), VolumeSize.width, VolumeSize.height);
	CopyParams.dstArray = pVolumeArray;
	CopyParams.extent	= VolumeSize;
	CopyParams.kind		= cudaMemcpyHostToDevice;

	CUDA_SAFE_CALL(cudaMemcpy3D(&CopyParams));

	gTexGradientMagnitude.normalized		= true;
	gTexGradientMagnitude.filterMode		= cudaFilterModePoint;
	gTexGradientMagnitude.addressMode[0]	= cudaAddressModeClamp;
	gTexGradientMagnitude.addressMode[1]	= cudaAddressModeClamp;

	cudaBindTextureToArray(gTexGradientMagnitude, pVolumeArray, VolumeChannelDesc);
}

void Render(const int& Type, CScene* pScene, CScene* pDevScene, unsigned int* pSeeds, CColorXyz* pDevEstFrameXyz, CColorXyz* pDevEstFrameBlurXyz, CColorXyz* pDevAccEstXyz, unsigned char* pDevEstRgbLdr, unsigned char* pDevEstRgbLdrDisp, int N)
{
	switch (Type)
	{
		case 0:
		{
			SingleScattering(pScene, pDevScene, pSeeds, pDevEstFrameXyz);
			break;
		}

		case 1:
		{
			MultipleScattering(pScene, pDevScene, pSeeds, pDevEstFrameXyz);
			break;
		}
	}

	BlurImageXyz(pDevEstFrameXyz, pDevEstFrameBlurXyz, CResolution2D(pScene->m_Camera.m_Film.m_Resolution.GetResX(), pScene->m_Camera.m_Film.m_Resolution.GetResY()), 1.3f);
  	ComputeEstimate(pScene->m_Camera.m_Film.m_Resolution.GetResX(), pScene->m_Camera.m_Film.m_Resolution.GetResY(), pDevEstFrameXyz, pDevAccEstXyz, N, pScene->m_Camera.m_Film.m_Exposure, pDevEstRgbLdr);
	Denoise(pScene, pDevScene, (CColorRgbLdr*)pDevEstRgbLdr, (CColorRgbLdr*)pDevEstRgbLdrDisp);
}