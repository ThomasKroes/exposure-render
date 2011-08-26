#pragma once

#include "Geometry.h"

class CLight;

class FIDELITY_RENDER_DLL CVolumePoint
{
public:
	float			m_T;				/*!< Parametric distance */
	Vec3f			m_P;				/*!< Position */
	short			m_D;				/*!< Density */
	CColorXyz		m_SigmaA;			/*!< Absorption coefficient */
	CColorXyz		m_SigmaS;			/*!< Scattering coefficient */
	float			m_Albedo;			/*!< Albedo */
	CColorXyz		m_Transmittance;	/*!< Transmittance */
	float			m_Ps;				/*!< Shader probability */

	// ToDo: Add description
	HOD CVolumePoint(void)
	{
		m_T				= 0.0f;
		m_P				= Vec3f(0.0f);
		m_D				= 0;
		m_SigmaA		= SPEC_WHITE;
		m_SigmaS		= SPEC_WHITE;
		m_Albedo		= 1.0f;
		m_Transmittance	= SPEC_WHITE;
		m_Ps			= 0.5f;		
	}

	// ToDo: Add description
	HOD ~CVolumePoint(void)
	{
	}

	// ToDo: Add description
	DEV CVolumePoint& CVolumePoint::operator=(const CVolumePoint& Other)
	{
		m_T					= Other.m_T;
		m_P					= Other.m_P;
		m_D					= Other.m_D;
		m_SigmaA			= Other.m_SigmaA;
		m_SigmaS			= Other.m_SigmaS;
		m_Albedo			= Other.m_Albedo;
		m_Transmittance		= Other.m_Transmittance;
		m_Ps				= Other.m_Ps;

		return *this;
	}
};