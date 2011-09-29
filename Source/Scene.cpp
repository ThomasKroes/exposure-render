
// Precompiled headers
#include "Stable.h"

#include "Scene.h"

CFilm& CFilm::operator=(const CFilm& Other)
{
	m_Resolution		= Other.m_Resolution;
	m_Screen[0][0]		= Other.m_Screen[0][0];
	m_Screen[0][1]		= Other.m_Screen[0][1];
	m_Screen[1][0]		= Other.m_Screen[1][0];
	m_Screen[1][1]		= Other.m_Screen[1][1];
	m_InvScreen			= Other.m_InvScreen;
	m_Iso				= Other.m_Iso;
	m_Exposure			= Other.m_Exposure;
	m_FStop				= Other.m_FStop;
	m_Gamma				= Other.m_Gamma;
	m_ToneMap			= Other.m_ToneMap;
	m_Bloom				= Other.m_Bloom;

	return *this;
}

CCamera& CCamera::operator=(const CCamera& Other)
{
	m_CameraType			= Other.m_CameraType;
	m_SceneBoundingBox		= Other.m_SceneBoundingBox;
	m_Hither				= Other.m_Hither;
	m_Yon					= Other.m_Yon;
	m_EnableClippingPlanes	= Other.m_EnableClippingPlanes;
	m_From					= Other.m_From;
	m_Target				= Other.m_Target;
	m_Up					= Other.m_Up;
	m_FovV					= Other.m_FovV;
	m_AreaPixel				= Other.m_AreaPixel;
	m_N						= Other.m_N;
	m_U						= Other.m_U;
	m_V						= Other.m_V;
	m_Film					= Other.m_Film;
	m_Focus					= Other.m_Focus;
	m_Aperture				= Other.m_Aperture;
	m_Dirty					= Other.m_Dirty;

	return *this;
}

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
	m_NoIterations(0),
	m_ExtinctionSize(),
	m_ShadingType(2),
	m_Spectral(false),
	m_StepSizeFactor(1.0f),
	m_StepSizeFactorShadow(1.0f),
	m_GradientDelta(1.0f),
	m_IOR(2.5f),
	m_GradientFactor(1.0f),
	m_GradMagMean(1.0f)
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
	m_Camera					= Other.m_Camera;
	m_Lighting					= Other.m_Lighting;
	m_Resolution				= Other.m_Resolution;
	m_DirtyFlags				= Other.m_DirtyFlags;
	m_Spacing					= Other.m_Spacing;
	m_Scale						= Other.m_Scale;
	m_BoundingBox				= Other.m_BoundingBox;
	m_PhaseG					= Other.m_PhaseG;
	m_TransferFunctions			= Other.m_TransferFunctions;
	m_MaxNoBounces				= Other.m_MaxNoBounces;
	m_IntensityRange			= Other.m_IntensityRange;
	m_KernelSize				= Other.m_KernelSize;
	m_SigmaMax					= Other.m_SigmaMax;
	m_DensityScale				= Other.m_DensityScale;
	m_MacrocellSize				= Other.m_MacrocellSize;
	m_DenoiseParams				= Other.m_DenoiseParams;
	m_NoIterations				= Other.m_NoIterations;
	m_ExtinctionSize			= Other.m_ExtinctionSize;
	m_ShadingType				= Other.m_ShadingType;
	m_Spectral					= Other.m_Spectral;
	m_StepSizeFactor			= Other.m_StepSizeFactor;
	m_StepSizeFactorShadow		= Other.m_StepSizeFactorShadow;
	m_GradientDelta				= Other.m_GradientDelta;
	m_GradientMagnitudeRange	= Other.m_GradientMagnitudeRange;
	m_IOR						= Other.m_IOR;
	m_GradientFactor			= Other.m_GradientFactor;
	m_GradMagMean				= Other.m_GradMagMean;

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

HO CDenoiseParams::CDenoiseParams(void)
{
	/*
	m_Enabled			= true;
	m_Noise				= 1.0f / (0.32f * 0.32f);
	m_LerpC				= 0.2f;
	m_WindowRadius		= 2.0f;
	m_WindowArea		= (2.0f * m_WindowRadius + 1.0f) * (2.0f * m_WindowRadius + 1.0f);
	m_InvWindowArea		= 1.0f / m_WindowArea;
	m_WeightThreshold	= 0.02f;
	m_LerpThreshold		= 0.79f;
	*/


	m_Enabled			= true;
	m_Noise				= 1.0f;// / (0.1f * 0.1f);
	m_LerpC				= 0.01f;
	m_WindowRadius		= 6.0f;
	m_WindowArea		= (2.0f * m_WindowRadius + 1.0f) * (2.0f * m_WindowRadius + 1.0f);
	m_InvWindowArea		= 1.0f / m_WindowArea;
	m_WeightThreshold	= 0.01f;
	m_LerpThreshold		= 0.01f;
	/**/
}
