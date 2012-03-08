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
#include "Framebuffer.cuh"
#include "Benchmark.cuh"

texture<unsigned short, cudaTextureType3D, cudaReadModeNormalizedFloat>		gTexIntensity;
texture<float, cudaTextureType1D, cudaReadModeElementType>					gTexOpacity;
texture<float4, cudaTextureType1D, cudaReadModeElementType>					gTexDiffuse;
texture<float4, cudaTextureType1D, cudaReadModeElementType>					gTexSpecular;
texture<float, cudaTextureType1D, cudaReadModeElementType>					gTexGlossiness;
texture<float4, cudaTextureType1D, cudaReadModeElementType>					gTexEmission;

cudaChannelFormatDesc gFloatChannelDesc = cudaCreateChannelDesc<float>();
cudaChannelFormatDesc gFloat4ChannelDesc = cudaCreateChannelDesc<float4>();

cudaArray* gpIntensity			= NULL;
cudaArray* gpOpacity			= NULL;
cudaArray* gpDiffuse			= NULL;
cudaArray* gpSpecular			= NULL;
cudaArray* gpGlossiness			= NULL;
cudaArray* gpEmission			= NULL;

CD ErVolume			gVolume;
CD ErCamera			gCamera;
CD ErLights			gLights;
CD ErClippers		gClippers;
CD ErReflectors		gReflectors;
CD ErDenoise		gDenoise;
CD ErScattering		gScattering;
CD ErBlur			gBlur;
CD ErRange			gOpacityRange;
CD ErRange			gDiffuseRange;
CD ErRange			gSpecularRange;
CD ErRange			gGlossinessRange;
CD ErRange			gEmissionRange;

#include "Blur.cuh"
#include "Denoise.cuh"
#include "Estimate.cuh"
#include "Utilities.cuh"
#include "SingleScattering.cuh"
#include "ToneMap.cuh"
#include "GradientMagnitude.cuh"

FrameBuffer FB;

void ErInitialize()
{
}

void ErDeinitialize()
{
	ErUnbindDensityBuffer();
	ErUnbindOpacity1D();
	ErUnbindDiffuse1D();
	ErUnbindSpecular1D();
	ErUnbindGlossiness1D();
	ErUnbindEmission1D();
}

void ErResize(int Size[2])
{
	FB.Resize(Resolution2i(Size));
}

void ErResetFrameBuffer()
{
	FB.Reset();
}

void ErBindIntensityBuffer(unsigned short* pBuffer, int Extent[3])
{
	ErUnbindDensityBuffer();

	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<unsigned short>();

	cudaExtent CudaExtent = make_cudaExtent(Extent[0], Extent[1], Extent[2]);

	cudaMalloc3DArray(&gpIntensity, &ChannelDesc, CudaExtent);

	cudaMemcpy3DParms CopyParams = {0};

	CopyParams.srcPtr		= make_cudaPitchedPtr(pBuffer, CudaExtent.width * sizeof(unsigned short), CudaExtent.width, CudaExtent.height);
	CopyParams.dstArray		= gpIntensity;
	CopyParams.extent		= CudaExtent;
	CopyParams.kind			= cudaMemcpyHostToDevice;
	
	cudaMemcpy3D(&CopyParams);

	gTexIntensity.normalized		= true;
	gTexIntensity.filterMode		= cudaFilterModeLinear;      
	gTexIntensity.addressMode[0]	= cudaAddressModeClamp;  
	gTexIntensity.addressMode[1]	= cudaAddressModeClamp;
  	gTexIntensity.addressMode[2]	= cudaAddressModeClamp;

	cudaBindTextureToArray(gTexIntensity, gpIntensity, ChannelDesc);
}

void ErUnbindDensityBuffer(void)
{
	cudaFreeArray(gpIntensity);
	gpIntensity = NULL;
	cudaUnbindTexture(gTexIntensity);
}

void ErBindOpacity1D(float Opacity[NO_GRADIENT_STEPS], float Range[2])
{
	ErRange Int;
	Int.Set(Range);

	cudaMemcpyToSymbol("gOpacityRange", &Int, sizeof(ErRange));

	gTexOpacity.normalized		= true;
	gTexOpacity.filterMode		= cudaFilterModeLinear;
	gTexOpacity.addressMode[0]	= cudaAddressModeClamp;

	if (gpOpacity == NULL)
		cudaMallocArray(&gpOpacity, &gFloatChannelDesc, NO_GRADIENT_STEPS, 1);

	cudaMemcpyToArray(gpOpacity, 0, 0, Opacity, NO_GRADIENT_STEPS * sizeof(float), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexOpacity, gpOpacity, gFloatChannelDesc);
}

void ErBindDiffuse1D(float Diffuse[3][NO_GRADIENT_STEPS], float Range[2])
{
	ErRange Int;
	Int.Set(Range);

	cudaMemcpyToSymbol("gDiffuseRange", &Int, sizeof(ErRange));

	gTexDiffuse.normalized		= true;
	gTexDiffuse.filterMode		= cudaFilterModeLinear;
	gTexDiffuse.addressMode[0]	= cudaAddressModeClamp;

	if (gpDiffuse == NULL)
		cudaMallocArray(&gpDiffuse, &gFloat4ChannelDesc, NO_GRADIENT_STEPS, 1);

	ColorXYZAf* pDiffuseXYZA = new ColorXYZAf[NO_GRADIENT_STEPS];

	for (int i = 0; i < NO_GRADIENT_STEPS; i++)
		pDiffuseXYZA[i].FromRGB(Diffuse[0][i], Diffuse[1][i], Diffuse[2][i]);

	cudaMemcpyToArray(gpDiffuse, 0, 0, pDiffuseXYZA, NO_GRADIENT_STEPS * sizeof(float4), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexDiffuse, gpDiffuse, gFloat4ChannelDesc);

	delete[] pDiffuseXYZA;
}

void ErBindSpecular1D(float Specular[3][NO_GRADIENT_STEPS], float Range[2])
{
	ErRange Int;
	Int.Set(Range);

	cudaMemcpyToSymbol("gSpecularRange", &Int, sizeof(ErRange));

	gTexSpecular.normalized		= true;
	gTexSpecular.filterMode		= cudaFilterModeLinear;
	gTexSpecular.addressMode[0]	= cudaAddressModeClamp;

	if (gpSpecular == NULL)
		cudaMallocArray(&gpSpecular, &gFloat4ChannelDesc, NO_GRADIENT_STEPS, 1);

	ColorXYZAf* pSpecularXYZA = new ColorXYZAf[NO_GRADIENT_STEPS];

	for (int i = 0; i < NO_GRADIENT_STEPS; i++)
		pSpecularXYZA[i].FromRGB(Specular[0][i], Specular[1][i], Specular[2][i]);

	cudaMemcpyToArray(gpSpecular, 0, 0, pSpecularXYZA, NO_GRADIENT_STEPS * sizeof(float4), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexSpecular, gpSpecular, gFloat4ChannelDesc);

	delete[] pSpecularXYZA;
}

void ErBindGlossiness1D(float Glossiness[NO_GRADIENT_STEPS], float Range[2])
{
	ErRange Int;
	Int.Set(Range);

	cudaMemcpyToSymbol("gGlossinessRange", &Int, sizeof(ErRange));

	gTexGlossiness.normalized		= true;
	gTexGlossiness.filterMode		= cudaFilterModeLinear;
	gTexGlossiness.addressMode[0]	= cudaAddressModeClamp;

	if (gpGlossiness == NULL)
		cudaMallocArray(&gpGlossiness, &gFloatChannelDesc, NO_GRADIENT_STEPS, 1);

	cudaMemcpyToArray(gpGlossiness, 0, 0, Glossiness, NO_GRADIENT_STEPS * sizeof(float),  cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexGlossiness, gpGlossiness, gFloatChannelDesc);
}

void ErBindEmission1D(float Emission[3][NO_GRADIENT_STEPS], float Range[2])
{
	ErRange Int;
	Int.Set(Range);

	cudaMemcpyToSymbol("gEmissionRange", &Int, sizeof(ErRange));

	gTexEmission.normalized		= true;
	gTexEmission.filterMode		= cudaFilterModeLinear;
	gTexEmission.addressMode[0]	= cudaAddressModeClamp;

	if (gpEmission == NULL)
		cudaMallocArray(&gpEmission, &gFloat4ChannelDesc, NO_GRADIENT_STEPS, 1);

	ColorXYZAf* pEmissionXYZA = new ColorXYZAf[NO_GRADIENT_STEPS];

	for (int i = 0; i < NO_GRADIENT_STEPS; i++)
		pEmissionXYZA[i].FromRGB(Emission[0][i], Emission[1][i], Emission[2][i]);

	cudaMemcpyToArray(gpEmission, 0, 0, pEmissionXYZA, NO_GRADIENT_STEPS * sizeof(float4),  cudaMemcpyHostToDevice);
	cudaBindTextureToArray(gTexEmission, gpEmission, gFloat4ChannelDesc);

	delete[] pEmissionXYZA;
}

void ErUnbindOpacity1D(void)
{
	cudaFreeArray(gpOpacity);
	gpOpacity = NULL;
	cudaUnbindTexture(gTexOpacity);
}

void ErUnbindDiffuse1D(void)
{
	cudaFreeArray(gpDiffuse);
	gpDiffuse = NULL;
	cudaUnbindTexture(gTexDiffuse);
}

void ErUnbindSpecular1D(void)
{
	cudaFreeArray(gpSpecular);
	gpSpecular	= NULL;
	cudaUnbindTexture(gTexSpecular);
}

void ErUnbindGlossiness1D(void)
{
	cudaFreeArray(gpGlossiness);
	gpGlossiness = NULL;
	cudaUnbindTexture(gTexGlossiness);
}

void ErUnbindEmission1D(void)
{
	cudaFreeArray(gpEmission);
	gpEmission	= NULL;
	cudaUnbindTexture(gTexEmission);
}

void ErBindVolume(ErVolume* pVolume)
{
	cudaMemcpyToSymbol("gVolume", pVolume, sizeof(ErVolume));
}

void ErBindCamera(ErCamera* pCamera)
{
	const Vec3f N = Normalize(ToVec3f(pCamera->Target) - ToVec3f(pCamera->Pos));
	const Vec3f U = Normalize(Cross(N, ToVec3f(pCamera->Up)));
	const Vec3f V = Normalize(Cross(N, U));

	pCamera->N[0] = N[0];
	pCamera->N[1] = N[1];
	pCamera->N[2] = N[2];
	pCamera->U[0] = U[0];
	pCamera->U[1] = U[1];
	pCamera->U[2] = U[2];
	pCamera->V[0] = V[0];
	pCamera->V[1] = V[1];
	pCamera->V[2] = V[2];

	float Scale = 0.0f;

	Scale = tanf((0.5f * pCamera->FOV / RAD_F));

	const float AspectRatio = (float)pCamera->FilmHeight / (float)pCamera->FilmWidth;

	if (AspectRatio > 1.0f)
	{
		pCamera->Screen[0][0] = -Scale;
		pCamera->Screen[0][1] = Scale;
		pCamera->Screen[1][0] = -Scale * AspectRatio;
		pCamera->Screen[1][1] = Scale * AspectRatio;
	}
	else
	{
		pCamera->Screen[0][0] = -Scale / AspectRatio;
		pCamera->Screen[0][1] = Scale / AspectRatio;
		pCamera->Screen[1][0] = -Scale;
		pCamera->Screen[1][1] = Scale;
	}

	pCamera->InvScreen[0] = (pCamera->Screen[0][1] - pCamera->Screen[0][0]) / (float)pCamera->FilmWidth;
	pCamera->InvScreen[1] = (pCamera->Screen[1][1] - pCamera->Screen[1][0]) / (float)pCamera->FilmHeight;

	cudaMemcpyToSymbol("gCamera", pCamera, sizeof(ErCamera));
}

void ErBindLights(ErLights* pLights)
{
	cudaMemcpyToSymbol("gLights", pLights, sizeof(ErLights));
}

void ErBindClippers(ErClippers* pClippers)
{
	cudaMemcpyToSymbol("gClippers", pClippers, sizeof(ErClippers));
}

void ErBindReflectors(ErReflectors* pReflectors)
{
	cudaMemcpyToSymbol("gReflectors", pReflectors, sizeof(ErReflectors));
}

void ErBindDenoise(ErDenoise* pDenoise)
{
	cudaMemcpyToSymbol("gDenoise", pDenoise, sizeof(ErDenoise));
}

void ErBindScattering(ErScattering* pScattering)
{
	cudaMemcpyToSymbol("gScattering", pScattering, sizeof(ErScattering));
}

void ErBindBlur(ErBlur* pBlur)
{
	cudaMemcpyToSymbol("gBlur", pBlur, sizeof(ErBlur));
}

void ErRenderEstimate()
{
	FrameBuffer* pDevFrameBuffer = NULL;
	cudaMalloc(&pDevFrameBuffer, sizeof(FrameBuffer));
	cudaMemcpy(pDevFrameBuffer, &FB, sizeof(FrameBuffer), cudaMemcpyHostToDevice);

	SingleScattering(pDevFrameBuffer, FB.Resolution[0], FB.Resolution[1]);
	BlurEstimate(pDevFrameBuffer, FB.Resolution[0], FB.Resolution[1]);
	ComputeEstimate(pDevFrameBuffer, FB.Resolution[0], FB.Resolution[1]);
	ToneMap(pDevFrameBuffer, FB.Resolution[0], FB.Resolution[1]);

	cudaFree(pDevFrameBuffer);
}

void ErGetEstimate(unsigned char* pData)
{
	cudaMemcpy(FB.HostDisplayEstimateRgbaLdr.GetPtr(), FB.CudaRunningEstimateRgbaLdr.GetPtr(), FB.CudaRunningEstimateRgbaLdr.GetSize(), cudaMemcpyDeviceToHost);
	memcpy(pData, FB.HostDisplayEstimateRgbaLdr.GetPtr(), FB.CudaRunningEstimateRgbaLdr.GetSize());
}

void ErGetFrameEstimate(unsigned char* pData)
{
	//cudaMemcpy(FB.HostDisplayEstimateRgbaLdr.GetPtr(), FB.CudaRunningEstimateRgbaLdr.GetPtr(), FB.CudaRunningEstimateRgbaLdr.GetSize(), cudaMemcpyDeviceToHost);
	//memcpy(pData, FB.HostDisplayEstimateRgbaLdr.GetPtr(), FB.CudaRunningEstimateRgbaLdr.GetSize());
}

void ErGetDepthBuffer(unsigned char* pData)
{
	//cudaMemcpy(FB.HostDisplayEstimateRgbaLdr.GetPtr(), FB.CudaRunningEstimateRgbaLdr.GetPtr(), FB.CudaRunningEstimateRgbaLdr.GetSize(), cudaMemcpyDeviceToHost);
	//memcpy(pData, FB.HostDisplayEstimateRgbaLdr.GetPtr(), FB.CudaRunningEstimateRgbaLdr.GetSize());
}

void ErRecordBenchmarkImage()
{
	cudaMemcpy(FB.BenchmarkEstimateRgbaLdr.GetPtr(), FB.CudaRunningEstimateRgbaLdr.GetPtr(), FB.CudaRunningEstimateRgbaLdr.GetSize(), cudaMemcpyDeviceToDevice);
}

void ErGetAverageNrmsError(float& AverageNrmsError)
{
	FrameBuffer* pDevFrameBuffer = NULL;
	cudaMalloc(&pDevFrameBuffer, sizeof(FrameBuffer));
	cudaMemcpy(pDevFrameBuffer, &FB, sizeof(FrameBuffer), cudaMemcpyHostToDevice);

	ComputeAverageNrmsError(FB, pDevFrameBuffer, FB.Resolution[0], FB.Resolution[1], AverageNrmsError);

	cudaFree(pDevFrameBuffer);
}

void ErGetMaximumGradientMagnitude(float& MaximumGradientMagnitude, int Extent[3])
{
	ComputeGradientMagnitudeVolume(Extent, MaximumGradientMagnitude);
}