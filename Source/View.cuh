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

#include "Buffer.cuh"

#include "Geometry.h"
#include "Scene.h"

class CCudaView
{
public:
	CCudaView(void) :
		m_Resolution(0, 0),
		m_RunningEstimateXyza(),
		m_FrameEstimateXyza(),
		m_FrameBlurXyza(),
		m_EstimateRgbaLdr(),
		m_DisplayEstimateRgbLdr(),
		m_RandomSeeds1(),
		m_RandomSeeds2()
	{
	}

	~CCudaView(void)
	{
		Free();
	}

	CCudaView::CCudaView(const CCudaView& Other)
	{
		*this = Other;
	}

	CCudaView& CCudaView::operator=(const CCudaView& Other)
	{
		m_Resolution			= Other.m_Resolution;
		m_RunningEstimateXyza	= Other.m_RunningEstimateXyza;
		m_FrameEstimateXyza		= Other.m_FrameEstimateXyza;
		m_FrameBlurXyza			= Other.m_FrameBlurXyza;
		m_EstimateRgbaLdr		= Other.m_EstimateRgbaLdr;
		m_DisplayEstimateRgbLdr	= Other.m_DisplayEstimateRgbLdr;
		m_RandomSeeds1			= Other.m_RandomSeeds1;
		m_RandomSeeds2			= Other.m_RandomSeeds2;

		return *this;
	}

	void Resize(const CResolution2D& Resolution)
	{
		if (m_Resolution == Resolution)
			return;

		m_Resolution = Resolution;

		m_RunningEstimateXyza.Resize(m_Resolution);
		m_FrameEstimateXyza.Resize(m_Resolution);
		m_FrameBlurXyza.Resize(m_Resolution);
		m_EstimateRgbaLdr.Resize(m_Resolution);
		m_DisplayEstimateRgbLdr.Resize(m_Resolution);
		m_RandomSeeds1.Resize(m_Resolution);
		m_RandomSeeds2.Resize(m_Resolution);
	}

	void Reset(void)
	{
//		m_RunningEstimateXyza.Reset();
		m_FrameEstimateXyza.Reset();
//		m_FrameBlurXyza.Reset();
		m_EstimateRgbaLdr.Reset();
		m_DisplayEstimateRgbLdr.Reset();
//		m_RandomSeeds1.Reset();
//		m_RandomSeeds2.Reset();
	}

	void Free(void)
	{
		m_RunningEstimateXyza.Free();
		m_FrameEstimateXyza.Free();
		m_FrameBlurXyza.Free();
		m_EstimateRgbaLdr.Free();
		m_DisplayEstimateRgbLdr.Free();
		m_RandomSeeds1.Free();
		m_RandomSeeds2.Free();

		m_Resolution.Set(Vec2i(0, 0));
	}

	HOD int GetWidth(void) const
	{
		return m_Resolution.GetResX();
	}

	HOD int GetHeight(void) const
	{
		return m_Resolution.GetResY();
	}

	CResolution2D						m_Resolution;
	CCudaBuffer2D<CColorXyza, true>		m_RunningEstimateXyza;
	CCudaBuffer2D<CColorXyza, true>		m_FrameEstimateXyza;
	CCudaBuffer2D<CColorXyza, true>		m_FrameBlurXyza;
	CCudaBuffer2D<ColorRGBAuc, true>	m_EstimateRgbaLdr;
	CCudaBuffer2D<ColorRGBuc, false>	m_DisplayEstimateRgbLdr;
	CCudaRandomBuffer2D					m_RandomSeeds1;
	CCudaRandomBuffer2D					m_RandomSeeds2;
};