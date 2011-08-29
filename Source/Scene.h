#pragma once

#include "Geometry.h"
#include "Random.h"
#include "Camera.h"
#include "Light.h"
#include "VolumePoint.h"
#include "Flags.h"
#include "Statistics.h"

#define MAX_NO_BINS 256

class CHistogram
{
public:
	CHistogram(void)
	{
		memset(m_Bins, 0, MAX_NO_BINS * sizeof(int));
		m_NoBins = MAX_NO_BINS;
	}

	CHistogram(const int* pBins, const int& NoBins)
	{
		SetBins(pBins, NoBins);
	}

	void SetBins(const int* pBins, const int& NoBins)
	{
		if (pBins == NULL)
			return;

		memcpy(m_Bins, pBins, NoBins * sizeof(int));
		m_NoBins = NoBins;
	}

private:
	int		m_Bins[MAX_NO_BINS];
	int		m_NoBins;
};

class CScene
{
public:
	CScene(void)
	{
		m_Light.m_Distance	= 1.5f;
		m_Light.m_Theta		= 0.0f;
		m_Light.m_Phi		= 0.0f;
		m_Light.m_Size		= 1.0f;

		m_TransferFunctions.m_Kd.m_NoNodes	= 4;
		m_TransferFunctions.m_Kd.m_P[0]		= 0.0f;
		m_TransferFunctions.m_Kd.m_P[1]		= 30.0f;
		m_TransferFunctions.m_Kd.m_P[2]		= 40.0f;
		m_TransferFunctions.m_Kd.m_P[3]		= 255.0f;

		m_TransferFunctions.m_Kd.m_C[0]		= CColorRgbHdr(0.01f);
		m_TransferFunctions.m_Kd.m_C[1]		= CColorRgbHdr(0.01f);
		m_TransferFunctions.m_Kd.m_C[2]		= CColorRgbHdr(1.0f);
		m_TransferFunctions.m_Kd.m_C[3]		= CColorRgbHdr(1.0f);

		m_TransferFunctions.m_Ks.m_NoNodes	= 4;
		m_TransferFunctions.m_Ks.m_P[0]		= 0.0f;
		m_TransferFunctions.m_Ks.m_P[1]		= 30.0f;
		m_TransferFunctions.m_Ks.m_P[2]		= 40.0f;
		m_TransferFunctions.m_Ks.m_P[3]		= 255.0f;

		m_TransferFunctions.m_Ks.m_C[0]		= CColorRgbHdr(0.01f);
		m_TransferFunctions.m_Ks.m_C[1]		= CColorRgbHdr(0.01f);
		m_TransferFunctions.m_Ks.m_C[2]		= CColorRgbHdr(1.0f);
		m_TransferFunctions.m_Ks.m_C[3]		= CColorRgbHdr(1.0f);

		m_TransferFunctions.m_Kt.m_NoNodes	= 4;
		m_TransferFunctions.m_Kt.m_P[0]		= 0.0f;
		m_TransferFunctions.m_Kt.m_P[1]		= 30.0f;
		m_TransferFunctions.m_Kt.m_P[2]		= 40.0f;
		m_TransferFunctions.m_Kt.m_P[3]		= 255.0f;

		m_TransferFunctions.m_Kt.m_C[0]		= CColorRgbHdr(0.01f);
		m_TransferFunctions.m_Kt.m_C[1]		= CColorRgbHdr(0.01f);
		m_TransferFunctions.m_Kt.m_C[2]		= CColorRgbHdr(1.0f);
		m_TransferFunctions.m_Kt.m_C[3]		= CColorRgbHdr(1.0f);
		
		

		m_PhaseG = 0.0f;

		m_MaxD = 255.0f;
		m_MaxNoBounces = 1;

		m_KernelSize.x = 32;
		m_KernelSize.y = 16;
	}

	virtual ~CScene(void)
	{
	}

	HOD CScene& operator = (const CScene& Other)			
	{
		m_Camera				= Other.m_Camera;
		m_Light					= Other.m_Light;
		m_Resolution			= Other.m_Resolution;
		m_DirtyFlags			= Other.m_DirtyFlags;
		m_Spacing				= Other.m_Spacing;
		m_Scale					= Other.m_Scale;
		m_BoundingBox			= Other.m_BoundingBox;
		m_PhaseG				= Other.m_PhaseG;
		m_MaxD					= Other.m_MaxD;
		m_TransferFunctions		= Other.m_TransferFunctions;
		m_MaxNoBounces			= Other.m_MaxNoBounces;
		m_MemorySize			= Other.m_MemorySize;
		m_NoVoxels				= Other.m_NoVoxels;
		m_IntensityRange		= Other.m_IntensityRange;
		m_KernelSize			= Other.m_KernelSize;
		m_FPS					= Other.m_FPS;

		return *this;
	}

	CCamera				m_Camera;
	CLight				m_Light;
	CResolution3D		m_Resolution;
	CFlags				m_DirtyFlags;
	Vec3f				m_Spacing;
	Vec3f				m_Scale;
	CBoundingBox		m_BoundingBox;
	float				m_PhaseG;
	float				m_MaxD;
	CTransferFunctions	m_TransferFunctions;
	int					m_MaxNoBounces;
	float				m_MemorySize;
	int					m_NoVoxels;
	CRange				m_IntensityRange;
	Vec2f				m_KernelSize;
	CEvent				m_FPS;
};

extern CScene* gpScene;