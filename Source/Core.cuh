#pragma once

#include "Geometry.h"
#include "Timing.h"
#include "Variance.h"
#include "CudaFrameBuffers.h"

#include <cuda_runtime.h>
#include <cutil.h>

class CScene;
class CVariance;

extern "C" void BindDensityBuffer(short* pBuffer, cudaExtent Extent);
extern "C" void BindGradientMagnitudeBuffer(short* pBuffer, cudaExtent Extent);
extern "C" void UnbindDensityBuffer(void);
extern "C" void UnbindGradientMagnitudeBuffer(void);
extern "C" void BindEstimateRgbLdr(CColorRgbaLdr* pBuffer, int Width, int Height);
extern "C" void Render(const int& Type, CScene* pScene, CScene* pDevScene, CCudaFrameBuffers& CudaFrameBuffers, int N, CVariance* pDevVariance, float* pVariance, CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage);