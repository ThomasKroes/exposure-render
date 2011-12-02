/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Core.cuh"

#include "Slicing.cuh"
#include "VolumeInfo.cuh"
#include "RenderInfo.cuh"
#include "Lighting.cuh"

texture<short, cudaTextureType3D, cudaReadModeNormalizedFloat>		gTexIntensity;
texture<short, cudaTextureType3D, cudaReadModeNormalizedFloat>		gTexGradientMagnitude;
texture<short, cudaTextureType3D, cudaReadModeNormalizedFloat>		gTexExtinction;
texture<float, cudaTextureType1D, cudaReadModeElementType>			gTexOpacity;
texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexDiffuse;
texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexSpecular;
texture<float, cudaTextureType1D, cudaReadModeElementType>			gTexGlossiness;
texture<float, cudaTextureType1D, cudaReadModeElementType>			gTexIOR;
texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexEmission;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat>		gTexRunningEstimateRgba;

cudaChannelFormatDesc gFloatChannelDesc = cudaCreateChannelDesc<float>();
cudaChannelFormatDesc gFloat4ChannelDesc = cudaCreateChannelDesc<float4>();

cudaArray* gpDensityArray			= NULL;
cudaArray* gpExtinction				= NULL;
cudaArray* gpGradientMagnitudeArray	= NULL;
cudaArray* gpOpacityArray			= NULL;
cudaArray* gpDiffuseArray			= NULL;
cudaArray* gpSpecularArray			= NULL;
cudaArray* gpGlossinessArray		= NULL;
cudaArray* gpIORArray				= NULL;
cudaArray* gpEmissionArray			= NULL;

CD Volume		gVolume;
CD Camera		gCamera;
CD Lighting		gLighting;
CD Slicing		gSlicing;
CD Denoise		gDenoise;
CD Scattering	gScattering;
CD Blur			gBlur;

#include "Blur.cuh"
#include "Denoise.cuh"
#include "Estimate.cuh"
#include "Utilities.cuh"
#include "SingleScattering.cuh"
#include "NearestIntersection.cuh"
#include "SpecularBloom.cuh"
#include "ToneMap.cuh"

void BindIntensityBuffer(short* pBuffer, cudaExtent Extent)
{
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<short>();

	HandleCudaError(cudaMalloc3DArray(&gpDensityArray, &ChannelDesc, Extent));

	cudaMemcpy3DParms CopyParams = {0};

	CopyParams.srcPtr	= make_cudaPitchedPtr(pBuffer, Extent.width * sizeof(short), Extent.width, Extent.height);
	CopyParams.dstArray	= gpDensityArray;
	CopyParams.extent	= Extent;
	CopyParams.kind		= cudaMemcpyHostToDevice;
	
	HandleCudaError(cudaMemcpy3D(&CopyParams));

	gTexIntensity.normalized		= true;
	gTexIntensity.filterMode		= cudaFilterModeLinear;      
	gTexIntensity.addressMode[0]	= cudaAddressModeClamp;  
	gTexIntensity.addressMode[1]	= cudaAddressModeClamp;
  	gTexIntensity.addressMode[2]	= cudaAddressModeClamp;

	HandleCudaError(cudaBindTextureToArray(gTexIntensity, gpDensityArray, ChannelDesc));
}

void BindExtinction(short* pBuffer, cudaExtent Extent)
{
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<short>();

	HandleCudaError(cudaMalloc3DArray(&gpExtinction, &ChannelDesc, Extent));

	cudaMemcpy3DParms CopyParams = {0};

	CopyParams.srcPtr	= make_cudaPitchedPtr(pBuffer, Extent.width * sizeof(short), Extent.width, Extent.height);
	CopyParams.dstArray	= gpExtinction;
	CopyParams.extent	= Extent;
	CopyParams.kind		= cudaMemcpyHostToDevice;
	
	HandleCudaError(cudaMemcpy3D(&CopyParams));

	gTexExtinction.normalized		= true;
	gTexExtinction.filterMode		= cudaFilterModePoint;      
	gTexExtinction.addressMode[0]	= cudaAddressModeClamp;  
	gTexExtinction.addressMode[1]	= cudaAddressModeClamp;
  	gTexExtinction.addressMode[2]	= cudaAddressModeClamp;

	HandleCudaError(cudaBindTextureToArray(gTexExtinction, gpExtinction, ChannelDesc));
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
	HandleCudaError(cudaUnbindTexture(gTexIntensity));
}

void UnbindGradientMagnitudeBuffer(void)
{
	HandleCudaError(cudaFreeArray(gpGradientMagnitudeArray));
	gpGradientMagnitudeArray = NULL;
	HandleCudaError(cudaUnbindTexture(gTexGradientMagnitude));
}

void BindTransferFunctions1D(float Opacity[128], float Diffuse[3][128], float Specular[3][128], float Glossiness[128], float IOR[128], float Emission[3][128], int N)
{
	// Opacity
	gTexOpacity.normalized		= true;
	gTexOpacity.filterMode		= cudaFilterModeLinear;
	gTexOpacity.addressMode[0]	= cudaAddressModeClamp;

	if (gpOpacityArray == NULL)
		HandleCudaError(cudaMallocArray(&gpOpacityArray, &gFloatChannelDesc, N, 1));

	HandleCudaError(cudaMemcpyToArray(gpOpacityArray, 0, 0, Opacity, N * sizeof(float), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexOpacity, gpOpacityArray, gFloatChannelDesc));

	// Diffuse
	gTexDiffuse.normalized		= true;
	gTexDiffuse.filterMode		= cudaFilterModeLinear;
	gTexDiffuse.addressMode[0]	= cudaAddressModeClamp;

	if (gpDiffuseArray == NULL)
		HandleCudaError(cudaMallocArray(&gpDiffuseArray, &gFloat4ChannelDesc, N, 1));

	ColorXYZAf* pDiffuseXYZA = new ColorXYZAf[N];

	for (int i = 0; i < N; i++)
		pDiffuseXYZA[i].FromRGB(Diffuse[0][i], Diffuse[1][i], Diffuse[2][i]);

	HandleCudaError(cudaMemcpyToArray(gpDiffuseArray, 0, 0, pDiffuseXYZA, N * sizeof(float4), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexDiffuse, gpDiffuseArray, gFloat4ChannelDesc));

	delete[] pDiffuseXYZA;

	// Specular
	gTexSpecular.normalized		= true;
	gTexSpecular.filterMode		= cudaFilterModeLinear;
	gTexSpecular.addressMode[0]	= cudaAddressModeClamp;

	if (gpSpecularArray == NULL)
		HandleCudaError(cudaMallocArray(&gpSpecularArray, &gFloat4ChannelDesc, N, 1));

	ColorXYZAf* pSpecularXYZA = new ColorXYZAf[N];

	for (int i = 0; i < N; i++)
		pSpecularXYZA[i].FromRGB(Specular[0][i], Specular[1][i], Specular[2][i]);

	HandleCudaError(cudaMemcpyToArray(gpSpecularArray, 0, 0, pSpecularXYZA, N * sizeof(float4), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexSpecular, gpSpecularArray, gFloat4ChannelDesc));

	delete[] pSpecularXYZA;

	// Glossiness
	gTexGlossiness.normalized		= true;
	gTexGlossiness.filterMode		= cudaFilterModeLinear;
	gTexGlossiness.addressMode[0]	= cudaAddressModeClamp;

	if (gpGlossinessArray == NULL)
		HandleCudaError(cudaMallocArray(&gpGlossinessArray, &gFloatChannelDesc, N, 1));

	HandleCudaError(cudaMemcpyToArray(gpGlossinessArray, 0, 0, Glossiness, N * sizeof(float),  cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexGlossiness, gpGlossinessArray, gFloatChannelDesc));

	// IOR
	gTexIOR.normalized		= true;
	gTexIOR.filterMode		= cudaFilterModeLinear;
	gTexIOR.addressMode[0]	= cudaAddressModeClamp;

	if (gpIORArray == NULL)
		HandleCudaError(cudaMallocArray(&gpIORArray, &gFloatChannelDesc, N, 1));

	HandleCudaError(cudaMemcpyToArray(gpIORArray, 0, 0, IOR, N * sizeof(float),  cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexIOR, gpIORArray, gFloatChannelDesc));

	// Emission
	gTexEmission.normalized		= true;
	gTexEmission.filterMode		= cudaFilterModeLinear;
	gTexEmission.addressMode[0]	= cudaAddressModeClamp;

	if (gpEmissionArray == NULL)
		HandleCudaError(cudaMallocArray(&gpEmissionArray, &gFloat4ChannelDesc, N, 1));

	ColorXYZAf* pEmissionXYZA = new ColorXYZAf[N];

	for (int i = 0; i < N; i++)
		pEmissionXYZA[i].FromRGB(Emission[0][i], Emission[1][i], Emission[2][i]);

	HandleCudaError(cudaMemcpyToArray(gpEmissionArray, 0, 0, pEmissionXYZA, N * sizeof(float4),  cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexEmission, gpEmissionArray, gFloat4ChannelDesc));

	delete[] pEmissionXYZA;
}

void UnbindTransferFunctions1D(void)
{
	HandleCudaError(cudaFreeArray(gpOpacityArray));
	HandleCudaError(cudaFreeArray(gpDiffuseArray));
	HandleCudaError(cudaFreeArray(gpSpecularArray));
	HandleCudaError(cudaFreeArray(gpGlossinessArray));
	HandleCudaError(cudaFreeArray(gpEmissionArray));

	gpOpacityArray		= NULL;
	gpDiffuseArray		= NULL;
	gpSpecularArray		= NULL;
	gpGlossinessArray	= NULL;
	gpEmissionArray		= NULL;

	HandleCudaError(cudaUnbindTexture(gTexOpacity));
	HandleCudaError(cudaUnbindTexture(gTexDiffuse));
	HandleCudaError(cudaUnbindTexture(gTexSpecular));
	HandleCudaError(cudaUnbindTexture(gTexGlossiness));
	HandleCudaError(cudaUnbindTexture(gTexEmission));
}

void RenderEstimate(Volume* pVolume, Camera* pCamera, Lighting* pLighting, Slicing* pSlicing, Denoise* pDenoise, Scattering* pScattering, Blur* pBlur, FrameBuffer* pFrameBuffer)
{
	HandleCudaError(cudaMemcpyToSymbol("gVolume", pVolume, sizeof(Volume)));
	HandleCudaError(cudaMemcpyToSymbol("gCamera", pCamera, sizeof(Camera)));
	HandleCudaError(cudaMemcpyToSymbol("gLighting", pLighting, sizeof(Lighting)));
	HandleCudaError(cudaMemcpyToSymbol("gSlicing", pSlicing, sizeof(Slicing)));
	HandleCudaError(cudaMemcpyToSymbol("gDenoise", pDenoise, sizeof(Denoise)));
	HandleCudaError(cudaMemcpyToSymbol("gScattering", pScattering, sizeof(Scattering)));
	HandleCudaError(cudaMemcpyToSymbol("gBlur", pBlur, sizeof(Blur)));
	
	FrameBuffer* pDevFrameBuffer = NULL;

	HandleCudaError(cudaMalloc(&pDevFrameBuffer, sizeof(FrameBuffer)));

	HandleCudaError(cudaMemcpy(pDevFrameBuffer, pFrameBuffer, sizeof(FrameBuffer), cudaMemcpyHostToDevice));

	const dim3 BlockDim(8, 8);
	const dim3 GridDim((int)ceilf((float)gCamera.m_FilmWidth / (float)BlockDim.x), (int)ceilf((float)gCamera.m_FilmHeight / (float)BlockDim.y));

	SingleScattering(pDevFrameBuffer, pCamera->m_FilmWidth, pCamera->m_FilmHeight);
	BlurEstimate(pDevFrameBuffer, pCamera->m_FilmWidth, pCamera->m_FilmHeight);
	ComputeEstimate(pDevFrameBuffer, pCamera->m_FilmWidth, pCamera->m_FilmHeight);
	ToneMap(pDevFrameBuffer, pCamera->m_FilmWidth, pCamera->m_FilmHeight);
//	ReduceNoise(pDevRenderInfo, pDevFrameBuffer, gCamera.m_FilmWidth, gCamera.m_FilmHeight);

	HandleCudaError(cudaFree(pDevFrameBuffer));
}