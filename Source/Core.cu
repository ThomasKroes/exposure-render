
#include "Core.cuh"

texture<float, 3, cudaReadModeElementType>	gTexDensity;
texture<float, 3, cudaReadModeElementType>	gTexExtinction;

#include "Blur.cuh"
#include "ComputeEstimate.cuh"
#include "Utilities.cuh"
#include "SingleScattering.cuh"
#include "MultipleScattering.cuh"

void BindDensityVolume(float* densityBuffer, cudaExtent densityBufferSize)
{
	/*cudaArray* gpDensity = NULL;

	// create 3D array
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&gpDensity, &ChannelDesc, Size);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams	= {0};
	copyParams.srcPtr				= make_cudaPitchedPtr(pDensityBuffer, Size.width * sizeof(float), Size.width, Size.height);
	copyParams.dstArray				= gpDensity;
	copyParams.extent				= Size;
	copyParams.kind					= cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	// Set texture parameters
	gTexDensity.normalized		= true;
	gTexDensity.filterMode		= cudaFilterModeLinear;      
	gTexDensity.addressMode[0]	= cudaAddressModeClamp;  
	gTexDensity.addressMode[1]	= cudaAddressModeClamp;
 	gTexDensity.addressMode[2]	= cudaAddressModeClamp;

	// Bind array to 3D texture
	cudaBindTextureToArray(gTexDensity, gpDensity, ChannelDesc);
	*/

	 cudaArray* densityArray;
    cudaChannelFormatDesc densityBufferChannelDesc = cudaCreateChannelDesc<float>();
    CUDA_SAFE_CALL(cudaMalloc3DArray(&densityArray, &densityBufferChannelDesc, densityBufferSize));
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)densityBuffer, densityBufferSize.width * sizeof(float), densityBufferSize.width, densityBufferSize.height);
    copyParams.dstArray = densityArray;
    copyParams.extent = densityBufferSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));

    gTexDensity.normalized = true;
    gTexDensity.filterMode = cudaFilterModeLinear;
    gTexDensity.addressMode[0] = cudaAddressModeClamp;
    gTexDensity.addressMode[1] = cudaAddressModeClamp;

    cudaBindTextureToArray( gTexDensity, densityArray, densityBufferChannelDesc);
}

void BindExtinctionVolume(float* extinction, cudaExtent extinctionSize)
{
/*
	cudaArray* gpExtinction = NULL;

	// create 3D array
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&gpExtinction, &ChannelDesc, Size);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams	= {0};
	copyParams.srcPtr				= make_cudaPitchedPtr(pExtinctionBuffer, Size.width * sizeof(float), Size.width, Size.height);
	copyParams.dstArray				= gpExtinction;
	copyParams.extent				= Size;
	copyParams.kind					= cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	// Set texture parameters
	gTexExtinction.normalized		= true;
	gTexExtinction.filterMode		= cudaFilterModePoint;      
	gTexExtinction.addressMode[0]	= cudaAddressModeClamp;  
	gTexExtinction.addressMode[1]	= cudaAddressModeClamp;
// 	gTexExtinction.addressMode[2]	= cudaAddressModeClamp;*/

	// Bind array to 3D texture
//	cudaBindTextureToArray(gTexExtinction, gpExtinction, ChannelDesc);
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

void Render(const int& Type, CScene* pScene, CScene* pDevScene, unsigned int* pSeeds, CColorXyz* pDevEstFrameXyz, CColorXyz* pDevEstFrameBlurXyz, CColorXyz* pDevAccEstXyz, unsigned char* pDevEstRgbLdr, int N)
{
	switch (Type)
	{
		case 0:
		{
			SingleScattering(pScene, pDevScene, pSeeds, pDevEstFrameXyz);
			BlurImageXyz(pDevEstFrameXyz, pDevEstFrameBlurXyz, CResolution2D(pScene->m_Camera.m_Film.m_Resolution.GetResX(), pScene->m_Camera.m_Film.m_Resolution.GetResY()), 1.3f);
  			ComputeEstimate(pScene->m_Camera.m_Film.m_Resolution.GetResX(), pScene->m_Camera.m_Film.m_Resolution.GetResY(), pDevEstFrameXyz, pDevAccEstXyz, N, 500.0f, pDevEstRgbLdr);
			
			break;
		}

		case 1:
		{
			break;
		}
	}
}