#pragma once

#include "Geometry.h"

#include <cuda_runtime.h>

extern "C" void BindDensityVolume(short* pDensityBuffer, cudaExtent Size);
extern "C" void BindExtinctionVolume(short* pExtinctionBuffer, cudaExtent Size);

extern "C" void Render(const int& Type, CScene* pScene, CScene* pDevScene, curandStateXORWOW_t* pDevRandomStates, CColorXyz* pDevEstFrameXyz);