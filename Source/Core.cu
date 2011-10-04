
#include "Core.cuh"

texture<short, 3, cudaReadModeNormalizedFloat>		gTexDensity;
texture<short, 3, cudaReadModeNormalizedFloat>		gTexGradientMagnitude;
texture<float, 3, cudaReadModeElementType>			gTexExtinction;
texture<uchar4, 2, cudaReadModeNormalizedFloat>		gTexEstimateRgbLdr;
texture<uchar4, 2, cudaReadModeNormalizedFloat>		gTexOpacity;
texture<uchar4, 2, cudaReadModeNormalizedFloat>		gTexDiffuse;
texture<uchar4, 2, cudaReadModeNormalizedFloat>		gTexSpecular;

cudaArray* gpDensityArray			= NULL;
cudaArray* gpGradientMagnitudeArray	= NULL;

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
	/**/

	CCudaTimer TmrDenoise;
	Denoise(pScene, pDevScene, CudaFrameBuffers.m_pDevEstRgbaLdr, CudaFrameBuffers.m_pDevRgbLdrDisp);
	HandleCudaError(cudaGetLastError());
	DenoiseImage.AddDuration(TmrDenoise.ElapsedTime());
	
}