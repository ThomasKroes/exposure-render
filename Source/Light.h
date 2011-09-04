#pragma once

#include "Geometry.h"

#define MAX_NO_LIGHTS 32

class CLight
{
public:
	float			m_Theta;
	float			m_Phi;
	float			m_Width;
	float			m_Height;
	float			m_Distance;
	float			m_Size;
	CColorRgbHdr	m_Color;

	CLight(void) :
		m_Theta(0.0f),
		m_Phi(0.0f),
		m_Distance(1.0f),
		m_Size(0.1f)
	{
	}

	HOD CLight& operator=(const CLight& Other)
	{
		m_Theta		= Other.m_Theta;
		m_Phi		= Other.m_Phi;
		m_Width		= Other.m_Width;
		m_Height	= Other.m_Height;
		m_Distance	= Other.m_Distance;
		m_Size		= Other.m_Size;
		m_Color		= Other.m_Color;

		return *this;
	}
};

class CLighting
{
public:
	CLighting(void) :
		m_NoLights(0)
	{
	}

	HOD CLighting& operator=(const CLighting& Other)
	{
		for (int i = 0; i < m_NoLights; i++)
		{
			m_Lights[i] = Other.m_Lights[i];
		}

		m_NoLights = Other.m_NoLights;

		return *this;
	}

	CLight	m_Lights[MAX_NO_LIGHTS];
	int		m_NoLights;
};