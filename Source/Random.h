#pragma once

#include "Geometry.h"

#include <curand_kernel.h>

class CCudaRNG
{
public:
	curandStateXORWOW_t		m_State;
	curandStateXORWOW_t*	m_pState;
	
	HOD CCudaRNG(curandStateXORWOW_t* pState)
	{
		m_State		= *pState;
		m_pState	= pState;
	}

	DEV void Init(curandStateXORWOW_t* pState)
	{
		m_State		= *pState;
		m_pState	= pState;
	}

	DEV ~CCudaRNG()
	{
		*m_pState = m_State;
	}

	DEV float Get1(void)
	{
		return curand_uniform(&m_State);
	}

	DEV Vec2f Get2(void)
	{
		return Vec2f(Get1(), Get1());
	}

	DEV Vec3f Get3(void)
	{
		return Vec3f(Get1(), Get1(), Get1());
	}
};