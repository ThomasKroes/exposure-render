#pragma once

#include "Geometry.h"

#include <curand_kernel.h>

/** 
* @struct Random
*
* @brief Random brief
*
* BXDF comments
* 
* @author Thomas Kroes
*/
class FIDELITY_RENDER_DLL CCudaRNG
{
public:
	curandStateXORWOW_t		m_State;
	curandStateXORWOW_t*	m_pState;
	
	// ToDo: Add description
	HOD CCudaRNG(curandStateXORWOW_t* pState)
	{
		m_State		= *pState;
		m_pState	= pState;
	}

	/**
		@brief Initializes the RNG
	*/
	DEV void Init(curandStateXORWOW_t* pState)
	{
		m_State		= *pState;
		m_pState	= pState;
	}

	DEV ~CCudaRNG()
	{
		*m_pState = m_State;
	}

	/**
		@brief Generates a single random float
		@return Random float
	*/
	DEV float Get1(void)
	{
		return curand_uniform(&m_State);
	}

	/**
		@brief Generates a two dimensional random float vector
		@return Two dimensional vector containing random floats
	*/
	DEV Vec2f Get2(void)
	{
		return Vec2f(Get1(), Get1());
	}

	/**
		@brief Generates a three dimensional random float vector
		@return Three dimensional vector containing random floats
	*/
	DEV Vec3f Get3(void)
	{
		return Vec3f(Get1(), Get1(), Get1());
	}
};

class FIDELITY_RENDER_DLL CRNG
{
public:
	CRNG(int seed = 5489UL)
	{
		mti = N+1; /* mti==N+1 means mt[N] is not initialized */
		Seed(seed);
	}

	void Seed(int seed) const;
	float RandomFloat() const;
	unsigned long RandomUInt() const;

private:
	static const int N = 624;
	mutable unsigned long mt[N]; /* the array for the state vector  */
	mutable int mti;
};