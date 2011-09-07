#pragma once

#include "Geometry.h"

#include "curand_kernel.h"

class CScene;

extern "C" void SetupRNG(CScene* pScene, CScene* pDevScene, curandStateXORWOW_t* pDevRandomStates);