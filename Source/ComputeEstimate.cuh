#pragma once

#include "Geometry.h"

extern "C" void ComputeEstimate(int Width, int Height, CColorXyz* pEstFrameXyz, CColorXyz* pAccEstXyz, float N, float Exposure, unsigned char* pPixels);