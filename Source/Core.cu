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

#include "General.cuh"

texture<float, cudaTextureType3D, cudaReadModeElementType>			gTexIntensity;
texture<float, cudaTextureType1D, cudaReadModeElementType>			gTexOpacity;
texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexDiffuse;
texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexSpecular;
texture<float, cudaTextureType1D, cudaReadModeElementType>			gTexGlossiness;
texture<float, cudaTextureType1D, cudaReadModeElementType>			gTexIOR;
texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexEmission;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat>		gTexRunningEstimateRgba;

cudaChannelFormatDesc gFloatChannelDesc = cudaCreateChannelDesc<float>();
cudaChannelFormatDesc gFloat4ChannelDesc = cudaCreateChannelDesc<float4>();

cudaArray* gpIntensity			= NULL;
cudaArray* gpExtinction			= NULL;
cudaArray* gpGradientMagnitude	= NULL;
cudaArray* gpOpacity			= NULL;
cudaArray* gpDiffuse			= NULL;
cudaArray* gpSpecular			= NULL;
cudaArray* gpGlossiness			= NULL;
cudaArray* gpIOR				= NULL;
cudaArray* gpEmission			= NULL;

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

void BindIntensityBuffer(float* pBuffer, cudaExtent Extent)
{
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float>();

	HandleCudaError(cudaMalloc3DArray(&gpIntensity, &ChannelDesc, Extent));

	cudaMemcpy3DParms CopyParams = {0};

	CopyParams.srcPtr	= make_cudaPitchedPtr(pBuffer, Extent.width * sizeof(float), Extent.width, Extent.height);
	CopyParams.dstArray	= gpIntensity;
	CopyParams.extent	= Extent;
	CopyParams.kind		= cudaMemcpyHostToDevice;
	
	HandleCudaError(cudaMemcpy3D(&CopyParams));

	gTexIntensity.normalized		= true;
	gTexIntensity.filterMode		= cudaFilterModeLinear;      
	gTexIntensity.addressMode[0]	= cudaAddressModeClamp;  
	gTexIntensity.addressMode[1]	= cudaAddressModeClamp;
  	gTexIntensity.addressMode[2]	= cudaAddressModeClamp;

	HandleCudaError(cudaBindTextureToArray(gTexIntensity, gpIntensity, ChannelDesc));
}

void UnbindDensityBuffer(void)
{
	HandleCudaError(cudaFreeArray(gpIntensity));
	gpIntensity = NULL;
	HandleCudaError(cudaUnbindTexture(gTexIntensity));
}

void BindTransferFunctions1D(float Opacity[128], float Diffuse[3][128], float Specular[3][128], float Glossiness[128], float IOR[128], float Emission[3][128], int N)
{
	// Opacity
	gTexOpacity.normalized		= true;
	gTexOpacity.filterMode		= cudaFilterModeLinear;
	gTexOpacity.addressMode[0]	= cudaAddressModeClamp;

	if (gpOpacity == NULL)
		HandleCudaError(cudaMallocArray(&gpOpacity, &gFloatChannelDesc, N, 1));

	HandleCudaError(cudaMemcpyToArray(gpOpacity, 0, 0, Opacity, N * sizeof(float), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexOpacity, gpOpacity, gFloatChannelDesc));

	// Diffuse
	gTexDiffuse.normalized		= true;
	gTexDiffuse.filterMode		= cudaFilterModeLinear;
	gTexDiffuse.addressMode[0]	= cudaAddressModeClamp;

	if (gpDiffuse == NULL)
		HandleCudaError(cudaMallocArray(&gpDiffuse, &gFloat4ChannelDesc, N, 1));

	ColorXYZAf* pDiffuseXYZA = new ColorXYZAf[N];

	for (int i = 0; i < N; i++)
		pDiffuseXYZA[i].FromRGB(Diffuse[0][i], Diffuse[1][i], Diffuse[2][i]);

	HandleCudaError(cudaMemcpyToArray(gpDiffuse, 0, 0, pDiffuseXYZA, N * sizeof(float4), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexDiffuse, gpDiffuse, gFloat4ChannelDesc));

	delete[] pDiffuseXYZA;

	// Specular
	gTexSpecular.normalized		= true;
	gTexSpecular.filterMode		= cudaFilterModeLinear;
	gTexSpecular.addressMode[0]	= cudaAddressModeClamp;

	if (gpSpecular == NULL)
		HandleCudaError(cudaMallocArray(&gpSpecular, &gFloat4ChannelDesc, N, 1));

	ColorXYZAf* pSpecularXYZA = new ColorXYZAf[N];

	for (int i = 0; i < N; i++)
		pSpecularXYZA[i].FromRGB(Specular[0][i], Specular[1][i], Specular[2][i]);

	HandleCudaError(cudaMemcpyToArray(gpSpecular, 0, 0, pSpecularXYZA, N * sizeof(float4), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexSpecular, gpSpecular, gFloat4ChannelDesc));

	delete[] pSpecularXYZA;

	// Glossiness
	gTexGlossiness.normalized		= true;
	gTexGlossiness.filterMode		= cudaFilterModeLinear;
	gTexGlossiness.addressMode[0]	= cudaAddressModeClamp;

	if (gpGlossiness == NULL)
		HandleCudaError(cudaMallocArray(&gpGlossiness, &gFloatChannelDesc, N, 1));

	HandleCudaError(cudaMemcpyToArray(gpGlossiness, 0, 0, Glossiness, N * sizeof(float),  cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexGlossiness, gpGlossiness, gFloatChannelDesc));

	// IOR
	gTexIOR.normalized		= true;
	gTexIOR.filterMode		= cudaFilterModeLinear;
	gTexIOR.addressMode[0]	= cudaAddressModeClamp;

	if (gpIOR == NULL)
		HandleCudaError(cudaMallocArray(&gpIOR, &gFloatChannelDesc, N, 1));

	HandleCudaError(cudaMemcpyToArray(gpIOR, 0, 0, IOR, N * sizeof(float),  cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexIOR, gpIOR, gFloatChannelDesc));

	// Emission
	gTexEmission.normalized		= true;
	gTexEmission.filterMode		= cudaFilterModeLinear;
	gTexEmission.addressMode[0]	= cudaAddressModeClamp;

	if (gpEmission == NULL)
		HandleCudaError(cudaMallocArray(&gpEmission, &gFloat4ChannelDesc, N, 1));

	ColorXYZAf* pEmissionXYZA = new ColorXYZAf[N];

	for (int i = 0; i < N; i++)
		pEmissionXYZA[i].FromRGB(Emission[0][i], Emission[1][i], Emission[2][i]);

	HandleCudaError(cudaMemcpyToArray(gpEmission, 0, 0, pEmissionXYZA, N * sizeof(float4),  cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexEmission, gpEmission, gFloat4ChannelDesc));

	delete[] pEmissionXYZA;
}

void UnbindTransferFunctions1D(void)
{
	HandleCudaError(cudaFreeArray(gpOpacity));
	HandleCudaError(cudaFreeArray(gpDiffuse));
	HandleCudaError(cudaFreeArray(gpSpecular));
	HandleCudaError(cudaFreeArray(gpGlossiness));
	HandleCudaError(cudaFreeArray(gpEmission));

	gpOpacity		= NULL;
	gpDiffuse		= NULL;
	gpSpecular		= NULL;
	gpGlossiness	= NULL;
	gpEmission		= NULL;

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