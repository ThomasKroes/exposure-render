#pragma once

#include "Geometry.h"
#include "RNG.h"

class CLightSample
{
public:
	Vec2f m_Pos;
	float m_Component;

	HOD CLightSample(void)
	{
		m_Pos	 	= Vec2f(0.0f);
		m_Component	= 0.0f;
	}

	HOD CLightSample& CLightSample::operator=(const CLightSample& Other)
	{
		m_Pos	 	= Other.m_Pos;
		m_Component = Other.m_Component;

		return *this;
	}

	DEV void LargeStep(CRNG& Rnd)
	{
		m_Pos		= Rnd.Get2();
		m_Component	= Rnd.Get1();
	}
};

class CBsdfSample
{
public:
	float	m_Component;
	Vec2f	m_Dir;

	HOD CBsdfSample(void)
	{
		m_Component = 0.0f;
		m_Dir 		= Vec2f(0.0f);
	}

	HOD CBsdfSample(const float& Component, const Vec2f& Dir)
	{
		m_Component = Component;
		m_Dir 		= Dir;
	}

	HOD CBsdfSample& CBsdfSample::operator=(const CBsdfSample& Other)
	{
		m_Component = Other.m_Component;
		m_Dir 		= Other.m_Dir;

		return *this;
	}

	DEV void LargeStep(CRNG& Rnd)
	{
		m_Component	= Rnd.Get1();
		m_Dir		= Rnd.Get2();
	}
};

class CLightingSample
{
public:
	CBsdfSample		m_BsdfSample;
	CLightSample 	m_LightSample;
	float			m_LightNum;

	HOD CLightingSample(void)
	{
		m_LightNum = 0.0f;
	}

	HOD CLightingSample& CLightingSample::operator=(const CLightingSample& Other)
	{
		m_BsdfSample	= Other.m_BsdfSample;
		m_LightNum		= Other.m_LightNum;
		m_LightSample	= Other.m_LightSample;

		return *this;
	}

	DEV void LargeStep(CRNG& Rnd)
	{
		m_BsdfSample.LargeStep(Rnd);
		m_LightSample.LargeStep(Rnd);

		m_LightNum = Rnd.Get1();
	}
};