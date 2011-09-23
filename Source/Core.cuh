#pragma once

#include "Geometry.h"
#include "Timing.h"

#include <cuda_runtime.h>
#include <cutil.h>

class CScene;

extern "C" void BindDensityVolume(short* pDensityBuffer, cudaExtent Size);
extern "C" void BindExtinctionVolume(float* pExtinctionBuffer, cudaExtent Size);
extern "C" void BindGradientMagnitudeVolume(short* pBuffer, cudaExtent VolumeSize);
extern "C" void BindEstimateRgbLdr(unsigned char* pBuffer, int Width, int Height);
extern "C" void Render(const int& Type, CScene* pScene, CScene* pDevScene, unsigned int* pSeeds, CColorXyz* pDevEstFrameXyz, CColorXyz* pDevEstFrameBlurXyz, CColorXyz* pDevAccEstXyz, unsigned char* pDevEstRgbLdr, unsigned char* pDevEstRgbLdrDisp, int N, CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage);