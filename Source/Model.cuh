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

class CCudaModel
{
public:
	CCudaModel(void) :
		m_Intensity(),
		m_GradientMagnitude()
	{
	}

	virtual ~CCudaModel(void)
	{
		Free();
	}

	CCudaModel::CCudaModel(const CCudaModel& Other)
	{
		*this = Other;
	}

	CCudaModel& CCudaModel::operator=(const CCudaModel& Other)
	{
		m_Intensity				= Other.m_Intensity;
		m_GradientMagnitude		= Other.m_GradientMagnitude;

		return *this;
	}

	void Resize(const CResolution3D& Resolution)
	{
		m_Intensity.Resize(Resolution);
		m_GradientMagnitude.Resize(Resolution);
	}

	void Free(void)
	{
		m_Intensity.Free();
		m_GradientMagnitude.Free();
	}

	CCudaBuffer3D<short>	m_Intensity;
	CCudaBuffer3D<short>	m_GradientMagnitude;
};