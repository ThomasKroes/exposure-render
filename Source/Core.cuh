/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

class FrameBuffer;

struct ErVolume;
struct ErCamera;
struct ErLights;
struct ErClippers;
struct ErReflectors;
struct ErDenoise;
struct ErScattering;
struct ErFiltering;

#define NO_GRADIENT_STEPS 256

__declspec(dllexport) void ErInitialize();
__declspec(dllexport) void ErDeinitialize();
__declspec(dllexport) void ErResize(int Size[2]);
__declspec(dllexport) void ErResetFrameBuffer();
__declspec(dllexport) void ErBindIntensityBuffer(unsigned short* pBuffer, int Extent[3]);
__declspec(dllexport) void ErBindExtinctionBuffer(unsigned short* pBuffer, int Extent[3]);
__declspec(dllexport) void ErUnbindDensityBuffer(void);
__declspec(dllexport) void ErBindOpacity1D(float Opacity[NO_GRADIENT_STEPS], float Range[2]);
__declspec(dllexport) void ErBindDiffuse1D(float Diffuse[3][NO_GRADIENT_STEPS], float Range[2]);
__declspec(dllexport) void ErBindSpecular1D(float Specular[3][NO_GRADIENT_STEPS], float Range[2]);
__declspec(dllexport) void ErBindGlossiness1D(float Glossiness[NO_GRADIENT_STEPS], float Range[2]);
__declspec(dllexport) void ErBindIor1D(float IOR[NO_GRADIENT_STEPS], float Range[2]);
__declspec(dllexport) void ErBindEmission1D(float Emission[3][NO_GRADIENT_STEPS], float Range[2]);
__declspec(dllexport) void ErUnbindOpacity1D(void);
__declspec(dllexport) void ErUnbindDiffuse1D(void);
__declspec(dllexport) void ErUnbindSpecular1D(void);
__declspec(dllexport) void ErUnbindGlossiness1D(void);
__declspec(dllexport) void ErUnbindIor1D(void);
__declspec(dllexport) void ErUnbindEmission1D(void);
__declspec(dllexport) void ErBindVolume(ErVolume* pVolume);
__declspec(dllexport) void ErBindCamera(ErCamera* pCamera);
__declspec(dllexport) void ErBindLights(ErLights* pLights);
__declspec(dllexport) void ErBindClippers(ErClippers* pClippers);
__declspec(dllexport) void ErBindReflectors(ErReflectors* pReflectors);
__declspec(dllexport) void ErBindDenoise(ErDenoise* pDenoise);
__declspec(dllexport) void ErBindScattering(ErScattering* pScattering);
__declspec(dllexport) void ErBindFiltering(ErFiltering* pFiltering);
__declspec(dllexport) void ErRenderEstimate();
__declspec(dllexport) void ErGetEstimate(unsigned char* pData);
__declspec(dllexport) void ErRecordBenchmarkImage();
__declspec(dllexport) void ErGetAverageNrmsError(float& AverageNrmsError);
__declspec(dllexport) void ErGetRunningVariance(float& RunningVariance);
__declspec(dllexport) void ErGetMaximumGradientMagnitude(float& MaximumGradientMagnitude, int Extent[3]);
__declspec(dllexport) void ErGetAutoFocusDistance(int FilmU, int FilmV, float& AutoFocusDistance);