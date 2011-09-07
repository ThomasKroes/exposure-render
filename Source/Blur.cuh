#pragma once

#include "Geometry.h"

extern "C" void BlurImageXyz(CColorXyz* pImage, CColorXyz* pTempImage, const CResolution2D& Resolution, const float& Radius);