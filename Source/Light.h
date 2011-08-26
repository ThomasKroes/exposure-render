#pragma once

#include "Dll.h"
#include "Defines.h"

class FIDELITY_RENDER_DLL CLight
{
public:
	float	m_Theta;
	float	m_Phi;
	float	m_Distance;
	float	m_Size;

	CLight(void) :
		m_Theta(0.0f),
		m_Phi(0.0f),
		m_Distance(1.0f),
		m_Size(0.1f)
	{
	}

	virtual ~CLight(void)
	{
	};

	// ToDo: Add description
	HOD CLight& operator=(const CLight& Other)
	{
		m_Theta		= Other.m_Theta;
		m_Phi		= Other.m_Phi;
		m_Distance	= Other.m_Distance;
		m_Size		= Other.m_Size;

		return *this;
	}
};