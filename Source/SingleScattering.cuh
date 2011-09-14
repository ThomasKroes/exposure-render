#pragma once

#include "Geometry.h"
// #include "Random.h"
// #include "Light.h"
// #include "VolumePoint.h"
#include "curand_kernel.h"

class CScene;

extern "C" void RenderVolume(CScene* pScene, CScene* pDevScene, curandStateXORWOW_t* pDevRandomStates, CColorXyz* pDevEstFrameXyz);