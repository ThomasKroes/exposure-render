/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Geometry.h"

#include "RNG.cuh"

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

class CBrdfSample
{
public:
	float	m_Component;
	Vec2f	m_Dir;

	HOD CBrdfSample(void)
	{
		m_Component = 0.0f;
		m_Dir 		= Vec2f(0.0f);
	}

	HOD CBrdfSample(const float& Component, const Vec2f& Dir)
	{
		m_Component = Component;
		m_Dir 		= Dir;
	}

	HOD CBrdfSample& CBrdfSample::operator=(const CBrdfSample& Other)
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
	CBrdfSample		m_BsdfSample;
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

class EXPOSURE_RENDER_DLL CCameraSample
{
public:
	Vec2f	m_ImageXY;
	Vec2f	m_LensUV;

	DEV CCameraSample(void)
	{
		m_ImageXY	= Vec2f(0.0f);
		m_LensUV	= Vec2f(0.0f);
	}

	DEV CCameraSample& CCameraSample::operator=(const CCameraSample& Other)
	{
		m_ImageXY	= Other.m_ImageXY;
		m_LensUV	= Other.m_LensUV;

		return *this;
	}

	DEV void LargeStep(Vec2f& ImageUV, Vec2f& LensUV, const int& X, const int& Y, const int& KernelSize)
	{
		m_ImageXY	= Vec2f(X + ImageUV.x, Y + ImageUV.y);
		m_LensUV	= LensUV;
	}
};