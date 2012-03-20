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

struct VolumeProperties;
struct Camera;
struct Lights;
struct Clippers;
struct Reflectors;
struct RenderSettings;
struct Filtering;
struct KernelTimings;
struct Textures;

#define NO_TF_STEPS 256

EXPOSURE_RENDER_DLL void Initialize();
EXPOSURE_RENDER_DLL void Deinitialize();
EXPOSURE_RENDER_DLL void Resize(int Size[2]);
EXPOSURE_RENDER_DLL void Reset();
EXPOSURE_RENDER_DLL void BindIntensityBuffer(unsigned short* pBuffer, int Extent[3]);
EXPOSURE_RENDER_DLL void BindOpacity1D(float Opacity[NO_TF_STEPS], float Range[2]);
EXPOSURE_RENDER_DLL void BindDiffuse1D(float Diffuse[3][NO_TF_STEPS], float Range[2]);
EXPOSURE_RENDER_DLL void BindSpecular1D(float Specular[3][NO_TF_STEPS], float Range[2]);
EXPOSURE_RENDER_DLL void BindGlossiness1D(float Glossiness[NO_TF_STEPS], float Range[2]);
EXPOSURE_RENDER_DLL void BindIor1D(float IOR[NO_TF_STEPS], float Range[2]);
EXPOSURE_RENDER_DLL void BindEmission1D(float Emission[3][NO_TF_STEPS], float Range[2]);
EXPOSURE_RENDER_DLL void BindVolumeProperties(VolumeProperties* pVolumeProperties);
EXPOSURE_RENDER_DLL void BindCamera(Camera* pCamera);
EXPOSURE_RENDER_DLL void BindLights(Lights* pLights);
EXPOSURE_RENDER_DLL void BindClippers(Clippers* pClippers);
EXPOSURE_RENDER_DLL void BindReflectors(Reflectors* pReflectors);
EXPOSURE_RENDER_DLL void BindRenderSettings(RenderSettings* pRenderSettings);
EXPOSURE_RENDER_DLL void BindFiltering(Filtering* pFiltering);
EXPOSURE_RENDER_DLL void BindTextures(Textures* pTextures);
EXPOSURE_RENDER_DLL void RenderEstimate();
EXPOSURE_RENDER_DLL void GetEstimate(unsigned char* pData);
EXPOSURE_RENDER_DLL void RecordBenchmarkImage();
EXPOSURE_RENDER_DLL void GetAverageNrmsError(float& AverageNrmsError);
EXPOSURE_RENDER_DLL void GetRunningVariance(float& RunningVariance);
EXPOSURE_RENDER_DLL void GetMaximumGradientMagnitude(float& MaximumGradientMagnitude, int Extent[3]);
EXPOSURE_RENDER_DLL void GetAutoFocusDistance(int FilmU, int FilmV, float& AutoFocusDistance);
EXPOSURE_RENDER_DLL void GetKernelTimings(KernelTimings* pKernelTimings);
EXPOSURE_RENDER_DLL void GetMemoryUsed(float& MemoryUsed);
EXPOSURE_RENDER_DLL void GetNoIterations(int& NoIterations);

}