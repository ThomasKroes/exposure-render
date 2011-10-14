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