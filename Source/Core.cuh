#pragma once

#include "Geometry.h"

#include <cuda_runtime.h>

extern "C" void BindDensityVolume(short* pDensityBuffer, cudaExtent Size);
extern "C" void BindExtinctionVolume(short* pExtinctionBuffer, cudaExtent Size);