#pragma once

#include "Geometry.h"
#include "Random.h"
#include "Camera.h"
#include "Light.h"
#include "VolumePoint.h"
#include "Flags.h"

#define MAX_NO_DURATIONS 100


class CEvent
{
public:
	CEvent(void) {};

	HO CEvent(const char* pName)
	{
#ifndef __CUDACC__
		sprintf_s(m_Name, "%s", pName);
#endif
		memset(m_Durations, 0, MAX_NO_DURATIONS * sizeof(float));

		m_NoDurations		= 0;
		m_FilteredDuration	= 0.0f;
	}

	virtual ~CEvent(void) {};

	HO CEvent& CEvent::operator=(const CEvent& Other)
	{
		strcpy_s(m_Name, Other.m_Name);

		for (int i = 0; i < MAX_NO_DURATIONS; i++)
		{
			m_Durations[i]	= Other.m_Durations[i];
		}

		m_NoDurations		= Other.m_NoDurations;
		m_FilteredDuration	= Other.m_FilteredDuration;

		return *this;
	}

	void AddDuration(const float& Duration)
	{
		float TempDurations[MAX_NO_DURATIONS];

		memcpy(TempDurations, m_Durations, MAX_NO_DURATIONS * sizeof(float));

		m_Durations[0] = Duration;

		//		m_FilteredDuration = Duration;
		//		return;

		float SumDuration = Duration;

		for (int i = 0; i < m_NoDurations - 1; i++)
		{
			m_Durations[i + 1] = TempDurations[i];
			SumDuration += TempDurations[i];
		}

		m_FilteredDuration = SumDuration / (float)m_NoDurations;

		m_NoDurations++;

		m_NoDurations = min(MAX_NO_DURATIONS, m_NoDurations);
	}

	char		m_Name[MAX_CHAR_SIZE];
	float		m_Durations[MAX_NO_DURATIONS];
	int			m_NoDurations;
	float		m_FilteredDuration;
};

class CScene
{
public:
	CScene(void)
	{
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
		m_Lighting				= Other.m_Lighting;
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
	CLighting			m_Lighting;
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