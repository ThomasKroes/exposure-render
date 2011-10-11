#pragma once

#include "Geometry.h"
#include "Flags.h"
#include "CameraInline.h"
#include "LightInline.h"

class EXPOSURE_RENDER_DLL CDenoiseParams
{
public:
	bool		m_Enabled;
	float		m_Noise;
	float		m_LerpC;
	float		m_WindowRadius;
	float		m_WindowArea;
	float		m_InvWindowArea;
	float		m_WeightThreshold;
	float		m_LerpThreshold;

public:
	HO CDenoiseParams(void);

	HOD CDenoiseParams& CDenoiseParams::operator=(const CDenoiseParams& Other)
	{
		m_Enabled			= Other.m_Enabled;
		m_Noise				= Other.m_Noise;
		m_LerpC				= Other.m_LerpC;
		m_WindowRadius		= Other.m_WindowRadius;
		m_WindowArea		= Other.m_WindowArea;
		m_InvWindowArea		= Other.m_InvWindowArea;
		m_WeightThreshold	= Other.m_WeightThreshold;
		m_LerpThreshold		= Other.m_LerpThreshold;

		return *this;
	}

	HOD void SetWindowRadius(const float& WindowRadius)
	{
		m_WindowRadius		= WindowRadius;
		m_WindowArea		= (2.0f * m_WindowRadius + 1.0f) * (2.0f * m_WindowRadius + 1.0f);
		m_InvWindowArea		= 1.0f / m_WindowArea;
	}
};

class EXPOSURE_RENDER_DLL CScene
{
public:
	CScene(void);
	CScene(const CScene& Other);
	CScene& operator = (const CScene& Other);

	CCamera				m_Camera;
	CLighting			m_Lighting;
	CResolution3D		m_Resolution;
	CFlags				m_DirtyFlags;
	Vec3f				m_Spacing;
	Vec3f				m_Scale;
	CBoundingBox		m_BoundingBox;
	float				m_PhaseG;
	CTransferFunctions	m_TransferFunctions;
	int					m_MaxNoBounces;
	CRange				m_IntensityRange;
	CRange				m_GradientMagnitudeRange;
	float				m_SigmaMax;
	float				m_DensityScale;
	CDenoiseParams		m_DenoiseParams;
	cudaExtent			m_ExtinctionSize;
	float				m_Variance;
	int					m_ShadingType;
	bool				m_Spectral;
	float				m_StepSizeFactor;
	float				m_StepSizeFactorShadow;
	float				m_GradientDelta;
	float				m_IOR;
	float				m_GradientFactor;
	float				m_GradMagMean;

	HOD int GetNoIterations(void) const					{ return m_NoIterations;			}
	HOD void SetNoIterations(const int& NoIterations)	{ m_NoIterations = NoIterations;	}

private:
	int					m_NoIterations;
};

extern CScene gScene;
