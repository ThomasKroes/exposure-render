
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
	m_MaxNoBounces(1),
	m_IntensityRange(),
	m_GradientMagnitudeRange(),
	m_KernelSize(16, 4),
	m_SigmaMax(0.0f),
	m_DensityScale(5000),
	m_MacrocellSize(32),
	m_DenoiseParams(),
	m_NoIterations(0)
{
}

CScene::CScene(const CScene& Other)
{
	*this = Other;
}

CScene::~CScene(void)
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
	m_SigmaMax			= Other.m_SigmaMax;
	m_DensityScale		= Other.m_DensityScale;
	m_MacrocellSize		= Other.m_MacrocellSize;
	m_DenoiseParams		= Other.m_DenoiseParams;
	m_NoIterations		= Other.m_NoIterations;

	return *this;
}

void CScene::PrintSelf(void)
{
// 	m_Camera.PrintSelf();
// 	m_Lighting.PrintSelf();

	printf("Volume Resolution: ");
 	m_Resolution.PrintSelf();
// 	m_DirtyFlags.PrintSelf();

	printf("Spacing: ");
 	m_Spacing.PrintSelf();

	printf("Scale: ");
 	m_Scale.PrintSelf();

	printf("Bounding Box: ");
	m_BoundingBox.PrintSelf();
// 	m_TransferFunctions.PrintSelf();

	printf("Intensity Range: ");
	m_IntensityRange.PrintSelf();

// 	m_KernelSize.PrintSelf();
}