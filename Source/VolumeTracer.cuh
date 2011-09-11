#pragma once

#include "Geometry.h"
#include "Random.h"
#include "Light.h"
#include "VolumePoint.h"

#include "curand_kernel.h"

class CScene;
class CRenderThread;

extern "C" void SetupRNG(CScene* pScene, CScene* pDevScene, curandStateXORWOW_t* pDevRandomStates);
extern "C" void	BindDensityVolume(float* pDensityBuffer, cudaExtent Size);
extern "C" void BindExtinctionVolume(float* pExtinctionBuffer, cudaExtent Size);
extern "C" void	UnbindVolumeData(void);
extern "C" void RenderVolume(CScene* pScene, CScene* pDevScene, curandStateXORWOW_t* pDevRandomStates, CColorXyz* pDevEstFrameXyz);