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

#include "General.cuh"

namespace ExposureRender
{

#ifdef _EXPORTING
	#define EXPOSURE_RENDER_DLL    __declspec(dllexport)
#else
	#define EXPOSURE_RENDER_DLL    __declspec(dllimport)
#endif

EXPOSURE_RENDER_DLL void InitializeTracer(int& TracerID);
EXPOSURE_RENDER_DLL void DeinitializeTracer(int TracerID);
EXPOSURE_RENDER_DLL void Resize(int TracerID, int Size[2]);
EXPOSURE_RENDER_DLL void Reset(int TracerID);
EXPOSURE_RENDER_DLL void BindOpacity1D(int TracerID, ErScalarTransferFunction1D Opacity1D);
EXPOSURE_RENDER_DLL void BindDiffuse1D(int TracerID, ErColorTransferFunction1D Diffuse1D);
EXPOSURE_RENDER_DLL void BindSpecular1D(int TracerID, ErColorTransferFunction1D Specular1D);
EXPOSURE_RENDER_DLL void BindGlossiness1D(int TracerID, ErScalarTransferFunction1D Glossiness1D);
EXPOSURE_RENDER_DLL void BindEmission1D(int TracerID, ErColorTransferFunction1D Emission1D);
EXPOSURE_RENDER_DLL void BindCamera(int TracerID, ErCamera Camera);
EXPOSURE_RENDER_DLL void BindVolume(int TracerID, int Resolution[3], float Spacing[3], unsigned short* pVoxels, bool NormalizeSize = false);
EXPOSURE_RENDER_DLL void BindLight(int TracerID, ErLight Light);
EXPOSURE_RENDER_DLL void UnbindLight(int TracerID, ErLight Light);
EXPOSURE_RENDER_DLL void BindObject(int TracerID, ErObject Object);
EXPOSURE_RENDER_DLL void UnbindObject(int TracerID, ErObject Object);
EXPOSURE_RENDER_DLL void BindClippingObject(int TracerID, ErClippingObject ClippingObject);
EXPOSURE_RENDER_DLL void UnbindClippingObject(int TracerID, ErClippingObject ClippingObject);
EXPOSURE_RENDER_DLL void BindTexture(int TracerID, ErTexture Texture);
EXPOSURE_RENDER_DLL void UnbindTexture(int TracerID, ErTexture Texture);
EXPOSURE_RENDER_DLL void BindRenderSettings(int TracerID, ErRenderSettings RenderSettings);
EXPOSURE_RENDER_DLL void BindFiltering(int TracerID, ErFiltering Filtering);
EXPOSURE_RENDER_DLL void RenderEstimate(int TracerID);
EXPOSURE_RENDER_DLL void GetEstimate(int TracerID, unsigned char* pData);
EXPOSURE_RENDER_DLL void GetAutoFocusDistance(int TracerID, int FilmU, int FilmV, float& AutoFocusDistance);
EXPOSURE_RENDER_DLL void GetKernelTimings(int TracerID, ErKernelTimings& KernelTimings);
EXPOSURE_RENDER_DLL void GetNoIterations(int TracerID, int& NoIterations);

}
