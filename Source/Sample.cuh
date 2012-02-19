/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Geometry.cuh"

#include "RNG.cuh"

struct LightSample
{
	Vec2f m_Pos;
	float m_Component;

	HOD LightSample(void)
	{
		m_Pos	 	= Vec2f(0.0f);
		m_Component	= 0.0f;
	}

	HOD LightSample& LightSample::operator=(const LightSample& Other)
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

struct BrdfSample
{
	float	m_Component;
	Vec2f	m_Dir;

	HOD BrdfSample(void)
	{
		m_Component = 0.0f;
		m_Dir 		= Vec2f(0.0f);
	}

	HOD BrdfSample(const float& Component, const Vec2f& Dir)
	{
		m_Component = Component;
		m_Dir 		= Dir;
	}

	HOD BrdfSample& BrdfSample::operator=(const BrdfSample& Other)
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

struct LightingSample
{
	BrdfSample		m_BsdfSample;
	LightSample 	m_LightSample;
	float			m_LightNum;

	HOD LightingSample(void)
	{
		m_LightNum = 0.0f;
	}

	HOD LightingSample& LightingSample::operator=(const LightingSample& Other)
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