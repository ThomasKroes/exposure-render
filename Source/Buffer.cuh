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

#include "Core.h"
#include "CudaUtilities.h"

template<class T, bool Pitched>
class CCudaBuffer2D
{
public:
	CCudaBuffer2D(void) :
		m_Resolution(0, 0),
		m_pData(NULL),
		m_Pitch(0)
	{
	}

	~CCudaBuffer2D(void)
	{
		Free();
	}

	void Resize(const CResolution2D& Resolution)
	{
		if (m_Resolution != Resolution)
			Free();

		m_Resolution = Resolution;

		if (GetNoElements() <= 0)
			return;

		if (Pitched)
			HandleCudaError(cudaMallocPitch((void**)&m_pData, &m_Pitch, GetWidth() * sizeof(T), GetHeight()));
		else
			HandleCudaError(cudaMalloc((void**)&m_pData, GetSize()));

		Reset();
	}

	void Reset(void)
	{
		if (GetSize() <= 0)
			return;

		HandleCudaError(cudaMemset(m_pData, 0, GetSize()));
	}

	void Free(void)
	{
		if (m_pData)
		{
			HandleCudaError(cudaFree(m_pData));
			m_pData = NULL;
		}
		
		m_Pitch	= 0;
		m_Resolution.Set(Vec2i(0, 0));
	}

	HOD int GetNoElements(void) const
	{
		return m_Resolution.GetNoElements();
	}

	HOD int GetSize(void) const
	{
		if (Pitched)
			return m_Resolution.GetResY() * m_Pitch;
		else
			return GetNoElements() * sizeof(T);
	}

	DEV T Get(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return T();

		if (Pitched)
			return m_pData[Y * (GetPitch() / sizeof(T)) + X];
		else
			return m_pData[Y * GetWidth() + X];
	}

	DEV T& GetRef(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return T();

		if (Pitched)
			return m_pData[Y * (GetPitch() / sizeof(T)) + X];
		else
			return m_pData[Y * GetWidth() + X];
	}

	HOD T* GetPtr(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return NULL;

		if (Pitched)
			return &m_pData[Y * (GetPitch() / sizeof(T)) + X];
		else
			return &m_pData[Y * GetWidth() + X];
	}

	DEV void Set(T Value, int X = 0, int Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return;

		if (Pitched)
			m_pData[Y * (GetPitch() / sizeof(T)) + X] = Value;
		else
			m_pData[Y * GetWidth() + X] = Value;
	}

	HOD int GetWidth(void) const
	{
		return m_Resolution.GetResX();
	}

	HOD int GetHeight(void) const
	{
		return m_Resolution.GetResY();
	}

	HOD int GetPitch(void) const
	{
		if (Pitched)
			return m_Pitch;
		else
			return GetWidth() * sizeof(T);
	}

protected:
	CResolution2D	m_Resolution;
	T*				m_pData;
	size_t			m_Pitch;
};

template<class T>
class CHostBuffer2D
{
public:
	CHostBuffer2D(void) :
		m_Resolution(0, 0),
		m_pData(NULL)
	{
	}

	virtual ~CHostBuffer2D(void)
	{
		Free();
	}

	void Resize(const CResolution2D& Resolution)
	{
		if (m_Resolution == Resolution)
			return;

		if (m_Resolution != Resolution)
			Free();

		m_Resolution = Resolution;

		if (GetNoElements() <= 0)
			return;

		m_pData = (T*)malloc(GetSize());

		Reset();
	}

	void Reset(void)
	{
		if (GetSize() <= 0)
			return;

		memset(m_pData, 0, GetSize());
	}

	void Free(void)
	{
		if (m_pData)
		{
			free(m_pData);
			m_pData = NULL;
		}
		
		m_Resolution.Set(Vec2i(0, 0));
	}

	HOD int GetNoElements(void) const
	{
		return m_Resolution.GetNoElements();
	}

	HOD int GetSize(void) const
	{
		return GetNoElements() * sizeof(T);
	}

	DEV T Get(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return T();

		return m_pData[Y * GetWidth() + X];
	}

	DEV T& GetRef(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return T();

		return m_pData[Y * GetWidth() + X];
	}

	HOD T* GetPtr(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return NULL;

		return &m_pData[Y * GetWidth() + X];
	}

	DEV void Set(T& Value, const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return;

		m_pData[Y * GetWidth() + X] = Value;
	}

	HOD int GetWidth(void) const
	{
		return m_Resolution.GetResX();
	}

	HOD int GetHeight(void) const
	{
		return m_Resolution.GetResY();
	}

protected:
	CResolution2D	m_Resolution;
	T*				m_pData;
};

class CCudaRandomBuffer2D : public CCudaBuffer2D<unsigned int, false>
{
public:
	void Resize(const CResolution2D& Resolution)
	{
		CCudaBuffer2D::Resize(Resolution);

		unsigned int* pSeeds = (unsigned int*)malloc(GetSize());

		memset(pSeeds, 0, GetSize());

		for (int i = 0; i < GetNoElements(); i++)
			pSeeds[i] = rand();

		HandleCudaError(cudaMemcpy(m_pData, pSeeds, GetSize(), cudaMemcpyHostToDevice));

		free(pSeeds);
	}
};

class FrameBuffer
{
public:
	FrameBuffer(void) :
		m_Resolution(0, 0),
		m_RunningEstimateXyza(),
		m_FrameEstimateXyza(),
		m_FrameBlurXyza(),
		m_EstimateRgbaLdr(),
		m_DisplayEstimateRgbLdr(),
		m_DisplayEstimateRgbaLdrHost(),
		m_RandomSeeds1(),
		m_RandomSeeds2()
	{
	}

	~FrameBuffer(void)
	{
		Free();
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
		m_DisplayEstimateRgbaLdrHost.Resize(m_Resolution);
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
		m_DisplayEstimateRgbaLdrHost.Reset();
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
		m_DisplayEstimateRgbaLdrHost.Free();
		m_RandomSeeds1.Free();
		m_RandomSeeds2.Free();

		m_Resolution.Set(Vec2i(0, 0));
	}

	DEV int GetWidth(void) const
	{
		return m_Resolution.GetResX();
	}

	DEV int GetHeight(void) const
	{
		return m_Resolution.GetResY();
	}

	CResolution2D						m_Resolution;
	CCudaBuffer2D<ColorXYZAf, false>	m_RunningEstimateXyza;
	CCudaBuffer2D<ColorXYZAf, false>	m_FrameEstimateXyza;
	CCudaBuffer2D<ColorXYZAf, false>	m_FrameBlurXyza;
	CCudaBuffer2D<ColorRGBAuc, false>	m_EstimateRgbaLdr;
	CCudaBuffer2D<ColorRGBuc, false>	m_DisplayEstimateRgbLdr;
	CHostBuffer2D<ColorRGBAuc>			m_DisplayEstimateRgbaLdrHost;
	CCudaRandomBuffer2D					m_RandomSeeds1;
	CCudaRandomBuffer2D					m_RandomSeeds2;
};