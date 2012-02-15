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
#include "Buffer.cuh"

texture<unsigned short, cudaTextureType3D, cudaReadModeNormalizedFloat>		gTexIntensity;
texture<float4, cudaTextureType1D, cudaReadModeElementType>					gTexEnvironmentGradient;
texture<float, cudaTextureType1D, cudaReadModeElementType>					gTexOpacity;
texture<float4, cudaTextureType1D, cudaReadModeElementType>					gTexDiffuse;
texture<float4, cudaTextureType1D, cudaReadModeElementType>					gTexSpecular;
texture<float, cudaTextureType1D, cudaReadModeElementType>					gTexGlossiness;
texture<float, cudaTextureType1D, cudaReadModeElementType>					gTexIor;
texture<float4, cudaTextureType1D, cudaReadModeElementType>					gTexEmission;

cudaChannelFormatDesc gFloatChannelDesc = cudaCreateChannelDesc<float>();
cudaChannelFormatDesc gFloat4ChannelDesc = cudaCreateChannelDesc<float4>();

cudaArray* gpIntensity			= NULL;
cudaArray* gEnvironmentGradient	= NULL;
cudaArray* gpOpacity			= NULL;
cudaArray* gpDiffuse			= NULL;
cudaArray* gpSpecular			= NULL;
cudaArray* gpGlossiness			= NULL;
cudaArray* gpIor				= NULL;
cudaArray* gpEmission			= NULL;

CD _Volume			gVolume;
CD _Camera			gCamera;
CD _Lighting		gLighting;
CD _Clipping		gClipping;
CD _Reflectors		gReflectors;
CD _Denoise			gDenoise;
CD _Scattering		gScattering;
CD _Blur			gBlur;

CD Interval			gOpacityRange;
CD Interval			gDiffuseRange;
CD Interval			gSpecularRange;
CD Interval			gGlossinessRange;
CD Interval			gIorRange;
CD Interval			gEmissionRange;

#include "Blur.cuh"
#include "Denoise.cuh"
#include "Estimate.cuh"
#include "Utilities.cuh"
#include "SingleScattering.cuh"
#include "NearestIntersection.cuh"
#include "SpecularBloom.cuh"
#include "ToneMap.cuh"

FrameBuffer FB;

void ErInitialize()
{
}

void ErDeinitialize()
{
}

void ErResize(int Size[2])
{
	FB.Resize(CResolution2D(Size[0], Size[1]));
}

void ErResetFrameBuffer()
{
	FB.Reset();
}

void ErBindIntensityBuffer(unsigned short* pBuffer, int Extent[3])
{
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<unsigned short>();

	cudaExtent CudaExtent = make_cudaExtent(Extent[0], Extent[1], Extent[2]);

	HandleCudaError(cudaMalloc3DArray(&gpIntensity, &ChannelDesc, CudaExtent));

	cudaMemcpy3DParms CopyParams = {0};

	CopyParams.srcPtr		= make_cudaPitchedPtr(pBuffer, CudaExtent.width * sizeof(unsigned short), CudaExtent.width, CudaExtent.height);
	CopyParams.dstArray		= gpIntensity;
	CopyParams.extent		= CudaExtent;
	CopyParams.kind			= cudaMemcpyHostToDevice;
	
	HandleCudaError(cudaMemcpy3D(&CopyParams));

	gTexIntensity.normalized		= true;
	gTexIntensity.filterMode		= cudaFilterModeLinear;      
	gTexIntensity.addressMode[0]	= cudaAddressModeClamp;  
	gTexIntensity.addressMode[1]	= cudaAddressModeClamp;
  	gTexIntensity.addressMode[2]	= cudaAddressModeClamp;

	HandleCudaError(cudaBindTextureToArray(gTexIntensity, gpIntensity, ChannelDesc));
}

void ErUnbindDensityBuffer(void)
{
	HandleCudaError(cudaFreeArray(gpIntensity));
	gpIntensity = NULL;
	HandleCudaError(cudaUnbindTexture(gTexIntensity));
}

void ErBindEnvironmentGradient(float EnvironmentGradient[3][NO_GRADIENT_STEPS])
{
	gTexEnvironmentGradient.normalized		= true;
	gTexEnvironmentGradient.filterMode		= cudaFilterModeLinear;
	gTexEnvironmentGradient.addressMode[0]	= cudaAddressModeClamp;

	if (gEnvironmentGradient == NULL)
		HandleCudaError(cudaMallocArray(&gEnvironmentGradient, &gFloat4ChannelDesc, NO_GRADIENT_STEPS, 1));

	ColorXYZAf* pEnvironmentGradientXYZA = new ColorXYZAf[NO_GRADIENT_STEPS];

	for (int i = 0; i < NO_GRADIENT_STEPS; i++)
		pEnvironmentGradientXYZA[i].FromRGB(EnvironmentGradient[0][i], EnvironmentGradient[1][i], EnvironmentGradient[2][i]);

	HandleCudaError(cudaMemcpyToArray(gEnvironmentGradient, 0, 0, pEnvironmentGradientXYZA, NO_GRADIENT_STEPS * sizeof(float4), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexEnvironmentGradient, gEnvironmentGradient, gFloat4ChannelDesc));

	delete[] pEnvironmentGradientXYZA;
}

void ErUnbindEnvironmentGradient(void)
{
	HandleCudaError(cudaFreeArray(gEnvironmentGradient));
	gEnvironmentGradient = NULL;
	HandleCudaError(cudaUnbindTexture(gTexEnvironmentGradient));
}

void ErBindOpacity1D(float Opacity[NO_GRADIENT_STEPS], float Range[2])
{
	Interval Int;
	Int.Set(Range);

	HandleCudaError(cudaMemcpyToSymbol("gOpacityRange", &Int, sizeof(Interval)));

	gTexOpacity.normalized		= true;
	gTexOpacity.filterMode		= cudaFilterModeLinear;
	gTexOpacity.addressMode[0]	= cudaAddressModeClamp;

	if (gpOpacity == NULL)
		HandleCudaError(cudaMallocArray(&gpOpacity, &gFloatChannelDesc, NO_GRADIENT_STEPS, 1));

	HandleCudaError(cudaMemcpyToArray(gpOpacity, 0, 0, Opacity, NO_GRADIENT_STEPS * sizeof(float), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexOpacity, gpOpacity, gFloatChannelDesc));
}

void ErBindDiffuse1D(float Diffuse[3][NO_GRADIENT_STEPS], float Range[2])
{
	Interval Int;
	Int.Set(Range);

	HandleCudaError(cudaMemcpyToSymbol("gDiffuseRange", &Int, sizeof(Interval)));

	gTexDiffuse.normalized		= true;
	gTexDiffuse.filterMode		= cudaFilterModeLinear;
	gTexDiffuse.addressMode[0]	= cudaAddressModeClamp;

	if (gpDiffuse == NULL)
		HandleCudaError(cudaMallocArray(&gpDiffuse, &gFloat4ChannelDesc, NO_GRADIENT_STEPS, 1));

	ColorXYZAf* pDiffuseXYZA = new ColorXYZAf[NO_GRADIENT_STEPS];

	for (int i = 0; i < NO_GRADIENT_STEPS; i++)
		pDiffuseXYZA[i].FromRGB(Diffuse[0][i], Diffuse[1][i], Diffuse[2][i]);

	HandleCudaError(cudaMemcpyToArray(gpDiffuse, 0, 0, pDiffuseXYZA, NO_GRADIENT_STEPS * sizeof(float4), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexDiffuse, gpDiffuse, gFloat4ChannelDesc));

	delete[] pDiffuseXYZA;
}

void ErBindSpecular1D(float Specular[3][NO_GRADIENT_STEPS], float Range[2])
{
	Interval Int;
	Int.Set(Range);

	HandleCudaError(cudaMemcpyToSymbol("gSpecularRange", &Int, sizeof(Interval)));

	gTexSpecular.normalized		= true;
	gTexSpecular.filterMode		= cudaFilterModeLinear;
	gTexSpecular.addressMode[0]	= cudaAddressModeClamp;

	if (gpSpecular == NULL)
		HandleCudaError(cudaMallocArray(&gpSpecular, &gFloat4ChannelDesc, NO_GRADIENT_STEPS, 1));

	ColorXYZAf* pSpecularXYZA = new ColorXYZAf[NO_GRADIENT_STEPS];

	for (int i = 0; i < NO_GRADIENT_STEPS; i++)
		pSpecularXYZA[i].FromRGB(Specular[0][i], Specular[1][i], Specular[2][i]);

	HandleCudaError(cudaMemcpyToArray(gpSpecular, 0, 0, pSpecularXYZA, NO_GRADIENT_STEPS * sizeof(float4), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexSpecular, gpSpecular, gFloat4ChannelDesc));

	delete[] pSpecularXYZA;
}

void ErBindGlossiness1D(float Glossiness[NO_GRADIENT_STEPS], float Range[2])
{
	Interval Int;
	Int.Set(Range);

	HandleCudaError(cudaMemcpyToSymbol("gGlossinessRange", &Int, sizeof(Interval)));

	gTexGlossiness.normalized		= true;
	gTexGlossiness.filterMode		= cudaFilterModeLinear;
	gTexGlossiness.addressMode[0]	= cudaAddressModeClamp;

	if (gpGlossiness == NULL)
		HandleCudaError(cudaMallocArray(&gpGlossiness, &gFloatChannelDesc, NO_GRADIENT_STEPS, 1));

	HandleCudaError(cudaMemcpyToArray(gpGlossiness, 0, 0, Glossiness, NO_GRADIENT_STEPS * sizeof(float),  cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexGlossiness, gpGlossiness, gFloatChannelDesc));
}

void ErBindIor1D(float Ior[NO_GRADIENT_STEPS], float Range[2])
{
	Interval Int;
	Int.Set(Range);

	HandleCudaError(cudaMemcpyToSymbol("gIorRange", &Int, sizeof(Interval)));

	gTexIor.normalized		= true;
	gTexIor.filterMode		= cudaFilterModeLinear;
	gTexIor.addressMode[0]	= cudaAddressModeClamp;

	if (gpIor == NULL)
		HandleCudaError(cudaMallocArray(&gpIor, &gFloatChannelDesc, NO_GRADIENT_STEPS, 1));

	HandleCudaError(cudaMemcpyToArray(gpIor, 0, 0, Ior, NO_GRADIENT_STEPS * sizeof(float),  cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexIor, gpIor, gFloatChannelDesc));
}

void ErBindEmission1D(float Emission[3][NO_GRADIENT_STEPS], float Range[2])
{
	Interval Int;
	Int.Set(Range);

	HandleCudaError(cudaMemcpyToSymbol("gEmissionRange", &Int, sizeof(Interval)));

	gTexEmission.normalized		= true;
	gTexEmission.filterMode		= cudaFilterModeLinear;
	gTexEmission.addressMode[0]	= cudaAddressModeClamp;

	if (gpEmission == NULL)
		HandleCudaError(cudaMallocArray(&gpEmission, &gFloat4ChannelDesc, NO_GRADIENT_STEPS, 1));

	ColorXYZAf* pEmissionXYZA = new ColorXYZAf[NO_GRADIENT_STEPS];

	for (int i = 0; i < NO_GRADIENT_STEPS; i++)
		pEmissionXYZA[i].FromRGB(Emission[0][i], Emission[1][i], Emission[2][i]);

	HandleCudaError(cudaMemcpyToArray(gpEmission, 0, 0, pEmissionXYZA, NO_GRADIENT_STEPS * sizeof(float4),  cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexEmission, gpEmission, gFloat4ChannelDesc));

	delete[] pEmissionXYZA;
}

void ErUnbindOpacity1D(void)
{
	HandleCudaError(cudaFreeArray(gpOpacity));
	gpOpacity = NULL;
	HandleCudaError(cudaUnbindTexture(gTexOpacity));
}

void ErUnbindDiffuse1D(void)
{
	HandleCudaError(cudaFreeArray(gpDiffuse));
	gpDiffuse = NULL;
	HandleCudaError(cudaUnbindTexture(gTexDiffuse));
}

void ErUnbindSpecular1D(void)
{
	HandleCudaError(cudaFreeArray(gpSpecular));
	gpSpecular	= NULL;
	HandleCudaError(cudaUnbindTexture(gTexSpecular));
}

void ErUnbindGlossiness1D(void)
{
	HandleCudaError(cudaFreeArray(gpGlossiness));
	gpGlossiness = NULL;
	HandleCudaError(cudaUnbindTexture(gTexGlossiness));
}

void ErUnbindIor1D(void)
{
	HandleCudaError(cudaFreeArray(gpIor));
	gpIor = NULL;
	HandleCudaError(cudaUnbindTexture(gTexIor));
}

void ErUnbindEmission1D(void)
{
	HandleCudaError(cudaFreeArray(gpEmission));
	gpEmission	= NULL;
	HandleCudaError(cudaUnbindTexture(gTexEmission));
}

void ErBindVolume(_Volume* pVolume)
{
	HandleCudaError(cudaMemcpyToSymbol("gVolume", pVolume, sizeof(_Volume)));
}

void ErBindCamera(_Camera* pCamera)
{
	const Vec3f N = Normalize(ToVec3f(pCamera->m_Target) - ToVec3f(pCamera->m_Pos));
	const Vec3f U = Normalize(Cross(N, ToVec3f(pCamera->m_Up)));
	const Vec3f V = Normalize(Cross(N, U));

	pCamera->m_N[0] = N[0];
	pCamera->m_N[1] = N[1];
	pCamera->m_N[2] = N[2];

	pCamera->m_U[0] = U[0];
	pCamera->m_U[1] = U[1];
	pCamera->m_U[2] = U[2];

	pCamera->m_V[0] = V[0];
	pCamera->m_V[1] = V[1];
	pCamera->m_V[2] = V[2];

	float Scale = 0.0f;

	Scale = tanf((0.5f * pCamera->m_FOV / RAD_F));

	const float AspectRatio = (float)pCamera->m_FilmHeight / (float)pCamera->m_FilmWidth;

	if (AspectRatio > 1.0f)
	{
		pCamera->m_Screen[0][0] = -Scale;
		pCamera->m_Screen[0][1] = Scale;
		pCamera->m_Screen[1][0] = -Scale * AspectRatio;
		pCamera->m_Screen[1][1] = Scale * AspectRatio;
	}
	else
	{
		pCamera->m_Screen[0][0] = -Scale / AspectRatio;
		pCamera->m_Screen[0][1] = Scale / AspectRatio;
		pCamera->m_Screen[1][0] = -Scale;
		pCamera->m_Screen[1][1] = Scale;
	}

	pCamera->m_InvScreen[0] = (pCamera->m_Screen[0][1] - pCamera->m_Screen[0][0]) / (float)pCamera->m_FilmWidth;
	pCamera->m_InvScreen[1] = (pCamera->m_Screen[1][1] - pCamera->m_Screen[1][0]) / (float)pCamera->m_FilmHeight;

	HandleCudaError(cudaMemcpyToSymbol("gCamera", pCamera, sizeof(_Camera)));
}

void ErBindLighting(_Lighting* pLighting)
{
	HandleCudaError(cudaMemcpyToSymbol("gLighting", pLighting, sizeof(_Lighting)));
}

void ErBindClipping(_Clipping* pClipping)
{
	HandleCudaError(cudaMemcpyToSymbol("gClipping", pClipping, sizeof(_Clipping)));
}

void ErBindReflectors(_Reflectors* pReflectors)
{
	HandleCudaError(cudaMemcpyToSymbol("gReflectors", pReflectors, sizeof(_Reflectors)));
}

void ErBindDenoise(_Denoise* pDenoise)
{
	HandleCudaError(cudaMemcpyToSymbol("gDenoise", pDenoise, sizeof(_Denoise)));
}

void ErBindScattering(_Scattering* pScattering)
{
	HandleCudaError(cudaMemcpyToSymbol("gScattering", pScattering, sizeof(_Scattering)));
}

void ErBindBlur(_Blur* pBlur)
{
	HandleCudaError(cudaMemcpyToSymbol("gBlur", pBlur, sizeof(_Blur)));
}

void ErRenderEstimate()
{
	FrameBuffer* pDevFrameBuffer = NULL;

	HandleCudaError(cudaMalloc(&pDevFrameBuffer, sizeof(FrameBuffer)));

//	HandleCudaError(cudaMemcpy(pDevFrameBuffer, pFrameBuffer, sizeof(FrameBuffer), cudaMemcpyHostToDevice));
	HandleCudaError(cudaMemcpy(pDevFrameBuffer, &FB, sizeof(FrameBuffer), cudaMemcpyHostToDevice));

	SingleScattering(pDevFrameBuffer, FB.m_Resolution[0], FB.m_Resolution[1]);
	BlurEstimate(pDevFrameBuffer, FB.m_Resolution[0], FB.m_Resolution[1]);
	ComputeEstimate(pDevFrameBuffer, FB.m_Resolution[0], FB.m_Resolution[1]);
	ToneMap(pDevFrameBuffer, FB.m_Resolution[0], FB.m_Resolution[1]);
//	ReduceNoise(pDevRenderInfo, pDevFrameBuffer, gCamera.m_FilmWidth, gCamera.m_FilmHeight);

	HandleCudaError(cudaFree(pDevFrameBuffer));
}

void ErGetRenderBuffer(unsigned char* pData)
{
	cudaMemcpy(FB.m_DisplayEstimateRgbaLdrHost.GetPtr(), FB.m_EstimateRgbaLdr.GetPtr(), FB.m_EstimateRgbaLdr.GetSize(), cudaMemcpyDeviceToHost);
	memcpy(pData, FB.m_DisplayEstimateRgbaLdrHost.GetPtr(), FB.m_EstimateRgbaLdr.GetSize());
}