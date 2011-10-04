#pragma once

#include "Geometry.h"

class CRNG
{
public:
	HOD CRNG(int* pSeed0, int* pSeed1)
	{
		m_pSeed0 = pSeed0;
		m_pSeed1 = pSeed1;
	}

	DEV float Get1(void)
	{
		*m_pSeed0 = 36969 * ((*m_pSeed0) & 65535) + ((*m_pSeed0) >> 16);
		*m_pSeed1 = 18000 * ((*m_pSeed1) & 65535) + ((*m_pSeed1) >> 16);

		int ires = ((*m_pSeed0) << 16) + (*m_pSeed1);

		union
		{
			float f;
			int ui;
		} res;

		res.ui = (ires & 0x007fffff) | 0x40000000;

		return (res.f - 2.f) / 2.f;
	}

	DEV Vec2f Get2(void)
	{
		return Vec2f(Get1(), Get1());
	}

	DEV Vec3f Get3(void)
	{
		return Vec3f(Get1(), Get1(), Get1());
	}

private:
	int*	m_pSeed0;
	int*	m_pSeed1;
};