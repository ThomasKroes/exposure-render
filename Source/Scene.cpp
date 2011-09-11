
// Precompiled headers
#include "Stable.h"

#include "Scene.h"

CScene::CScene(void) :
	m_Camera(),
	m_Lighting(),
	m_Resolution(),
	m_DirtyFlags(),
	m_Spacing(),
	m_Scale(),
	m_BoundingBox(),
	m_PhaseG(0.0f),
	m_TransferFunctions(),
	m_MaxNoBounces(3),
	m_IntensityRange(),
	m_KernelSize(16, 8),
	m_SigmaMax(5000)
{
}

HOD CScene& CScene::operator=(const CScene& Other)
{
	m_Camera			= Other.m_Camera;
	m_Lighting			= Other.m_Lighting;
	m_Resolution		= Other.m_Resolution;
	m_DirtyFlags		= Other.m_DirtyFlags;
	m_Spacing			= Other.m_Spacing;
	m_Scale				= Other.m_Scale;
	m_BoundingBox		= Other.m_BoundingBox;
	m_PhaseG			= Other.m_PhaseG;
	m_TransferFunctions	= Other.m_TransferFunctions;
	m_MaxNoBounces		= Other.m_MaxNoBounces;
	m_IntensityRange	= Other.m_IntensityRange;
	m_KernelSize		= Other.m_KernelSize;
	m_SigmaMax				= Other.m_SigmaMax;

	return *this;
}

void CScene::Set(void)
{
//	printf("sad");
}
