#pragma once

#include "Geometry.h"

#include <cuda_runtime.h>
#include <cutil.h>

class CScene;

extern "C" void BindDensityVolume(float* pDensityBuffer, cudaExtent Size);
extern "C" void BindExtinctionVolume(float* pExtinctionBuffer, cudaExtent Size);
extern "C" void Render(const int& Type, CScene* pScene, CScene* pDevScene, unsigned int* pSeeds, CColorXyz* pDevEstFrameXyz, CColorXyz* pDevEstFrameBlurXyz, CColorXyz* pDevAccEstXyz, unsigned char* pDevEstRgbLdr, int N);