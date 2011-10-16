/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Geometry.h"
#include "Timing.h"
#include "Scene.h"

class CScene;
class CVariance;

extern "C" void BindDensityBuffer(short* pBuffer, cudaExtent Extent);
extern "C" void BindGradientMagnitudeBuffer(short* pBuffer, cudaExtent Extent);
extern "C" void UnbindDensityBuffer(void);
extern "C" void UnbindGradientMagnitudeBuffer(void);
extern "C" void BindRenderCanvasView(const CResolution2D& Resolution);
extern "C" void ResetRenderCanvasView(void);
extern "C" void FreeRenderCanvasView(void);
extern "C" unsigned char* GetDisplayEstimate(void);
extern "C" void BindTransferFunctionOpacity(CTransferFunction& TransferFunctionOpacity);
extern "C" void BindTransferFunctionDiffuse(CTransferFunction& TransferFunctionDiffuse);
extern "C" void BindTransferFunctionSpecular(CTransferFunction& TransferFunctionSpecular);
extern "C" void BindTransferFunctionRoughness(CTransferFunction& TransferFunctionRoughness);
extern "C" void BindTransferFunctionEmission(CTransferFunction& TransferFunctionEmission);
extern "C" void UnbindTransferFunctionOpacity(void);
extern "C" void UnbindTransferFunctionDiffuse(void);
extern "C" void UnbindTransferFunctionSpecular(void);
extern "C" void UnbindTransferFunctionRoughness(void);
extern "C" void UnbindTransferFunctionEmission(void);
extern "C" void BindConstants(CScene* pScene);
extern "C" void Render(const int& Type, CScene& Scene, CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage);