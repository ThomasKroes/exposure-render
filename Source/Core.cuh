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

namespace ExposureRender
{

#ifdef _EXPORTING
	#define EXPOSURE_RENDER_DLL    __declspec(dllexport)
#else
	#define EXPOSURE_RENDER_DLL    __declspec(dllimport)
#endif

struct ErVolume;
struct ErCamera;
struct ErLights;
struct ErClippers;
struct ErReflectors;
struct ErDenoise;
struct ErScattering;
struct ErFiltering;
struct ErKernelTimings;

#define NO_TF_STEPS 256

EXPOSURE_RENDER_DLL void ErInitialize();
EXPOSURE_RENDER_DLL void ErDeinitialize();
EXPOSURE_RENDER_DLL void ErResize(int Size[2]);
EXPOSURE_RENDER_DLL void ErReset();
EXPOSURE_RENDER_DLL void ErBindIntensityBuffer(unsigned short* pBuffer, int Extent[3]);
EXPOSURE_RENDER_DLL void ErBindOpacity1D(float Opacity[NO_TF_STEPS], float Range[2]);
EXPOSURE_RENDER_DLL void ErBindDiffuse1D(float Diffuse[3][NO_TF_STEPS], float Range[2]);
EXPOSURE_RENDER_DLL void ErBindSpecular1D(float Specular[3][NO_TF_STEPS], float Range[2]);
EXPOSURE_RENDER_DLL void ErBindGlossiness1D(float Glossiness[NO_TF_STEPS], float Range[2]);
EXPOSURE_RENDER_DLL void ErBindIor1D(float IOR[NO_TF_STEPS], float Range[2]);
EXPOSURE_RENDER_DLL void ErBindEmission1D(float Emission[3][NO_TF_STEPS], float Range[2]);
EXPOSURE_RENDER_DLL void ErUnbindOpacity1D(void);
EXPOSURE_RENDER_DLL void ErUnbindDiffuse1D(void);
EXPOSURE_RENDER_DLL void ErUnbindSpecular1D(void);
EXPOSURE_RENDER_DLL void ErUnbindGlossiness1D(void);
EXPOSURE_RENDER_DLL void ErUnbindIor1D(void);
EXPOSURE_RENDER_DLL void ErUnbindEmission1D(void);
EXPOSURE_RENDER_DLL void ErBindVolume(ErVolume* pVolume);
EXPOSURE_RENDER_DLL void ErBindCamera(ErCamera* pCamera);
EXPOSURE_RENDER_DLL void ErBindLights(ErLights* pLights);
EXPOSURE_RENDER_DLL void ErBindClippers(ErClippers* pClippers);
EXPOSURE_RENDER_DLL void ErBindReflectors(ErReflectors* pReflectors);
EXPOSURE_RENDER_DLL void ErBindScattering(ErScattering* pScattering);
EXPOSURE_RENDER_DLL void ErBindFiltering(ErFiltering* pFiltering);
EXPOSURE_RENDER_DLL void ErRenderEstimate();
EXPOSURE_RENDER_DLL void ErGetEstimate(unsigned char* pData);
EXPOSURE_RENDER_DLL void ErRecordBenchmarkImage();
EXPOSURE_RENDER_DLL void ErGetAverageNrmsError(float& AverageNrmsError);
EXPOSURE_RENDER_DLL void ErGetRunningVariance(float& RunningVariance);
EXPOSURE_RENDER_DLL void ErGetMaximumGradientMagnitude(float& MaximumGradientMagnitude, int Extent[3]);
EXPOSURE_RENDER_DLL void ErGetAutoFocusDistance(int FilmU, int FilmV, float& AutoFocusDistance);
EXPOSURE_RENDER_DLL void ErGetKernelTimings(ErKernelTimings* pKernelTimings);
EXPOSURE_RENDER_DLL void ErGetMemoryUsed(float& MemoryUsed);
EXPOSURE_RENDER_DLL void ErGetNoIterations(int& NoIterations);

}