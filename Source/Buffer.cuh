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

#include "CudaUtilities.h"

#include "Geometry.cuh"

template<class T, bool Pitched>
class CCudaBuffer2D
{
public:
	CCudaBuffer2D(void) :
		m_Resolution(),
		m_pData(NULL),
		m_Pitch(0)
	{
	}

	~CCudaBuffer2D(void)
	{
		Free();
	}

	HOST void Resize(Resolution2i Resolution)
	{
		if (m_Resolution != Resolution)
			Free();

		m_Resolution = Resolution;

		if (GetNoElements() <= 0)
			return;

		if (Pitched)
			cudaMallocPitch((void**)&m_pData, &m_Pitch, GetWidth() * sizeof(T), GetHeight());
		else
			cudaMalloc((void**)&m_pData, GetSize());

		Reset();
	}

	HOST void Reset(void)
	{
		if (GetSize() <= 0)
			return;

		cudaMemset(m_pData, 0, GetSize());
	}

	HOST void Free(void)
	{
		if (m_pData)
		{
			cudaFree(m_pData);
			m_pData = NULL;
		}
		
		m_Pitch	= 0;
		m_Resolution = Resolution2i();
	}

	HOST_DEVICE int GetNoElements(void) const
	{
		return m_Resolution.GetNoElements();
	}

	HOST_DEVICE int GetSize(void) const
	{
		if (Pitched)
			return m_Resolution[1] * m_Pitch;
		else
			return GetNoElements() * sizeof(T);
	}

	HOST_DEVICE T Get(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return T();

		if (Pitched)
			return m_pData[Y * (GetPitch() / sizeof(T)) + X];
		else
			return m_pData[Y * GetWidth() + X];
	}

	HOST_DEVICE T& GetRef(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return T();

		if (Pitched)
			return m_pData[Y * (GetPitch() / sizeof(T)) + X];
		else
			return m_pData[Y * GetWidth() + X];
	}

	HOST_DEVICE T* GetPtr(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return NULL;

		if (Pitched)
			return &m_pData[Y * (GetPitch() / sizeof(T)) + X];
		else
			return &m_pData[Y * GetWidth() + X];
	}

	HOST_DEVICE void Set(T Value, int X = 0, int Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return;

		if (Pitched)
			m_pData[Y * (GetPitch() / sizeof(T)) + X] = Value;
		else
			m_pData[Y * GetWidth() + X] = Value;
	}

	HOST_DEVICE int GetWidth(void) const
	{
		return m_Resolution[0];
	}

	HOST_DEVICE int GetHeight(void) const
	{
		return m_Resolution[1];
	}

	HOST_DEVICE int GetPitch(void) const
	{
		if (Pitched)
			return m_Pitch;
		else
			return GetWidth() * sizeof(T);
	}

protected:
	Resolution2i		m_Resolution;
	T*					m_pData;
	size_t				m_Pitch;
};

template<class T>
class CHostBuffer2D
{
public:
	CHostBuffer2D(void) :
		m_Resolution(),
		m_pData(NULL)
	{
	}

	virtual ~CHostBuffer2D(void)
	{
		Free();
	}

	void Resize(Resolution2i Resolution)
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
		
		m_Resolution = Resolution2i();
	}

	HOST_DEVICE int GetNoElements(void) const
	{
		return m_Resolution.GetNoElements();
	}

	HOST_DEVICE int GetSize(void) const
	{
		return GetNoElements() * sizeof(T);
	}

	HOST_DEVICE T Get(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return T();

		return m_pData[Y * GetWidth() + X];
	}

	HOST_DEVICE T& GetRef(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return T();

		return m_pData[Y * GetWidth() + X];
	}

	HOST_DEVICE T* GetPtr(const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return NULL;

		return &m_pData[Y * GetWidth() + X];
	}

	HOST_DEVICE void Set(T& Value, const int& X = 0, const int& Y = 0)
	{
		if (X > GetWidth() || Y > GetHeight())
			return;

		m_pData[Y * GetWidth() + X] = Value;
	}

	HOST_DEVICE int GetWidth(void) const
	{
		return m_Resolution[0];
	}

	HOST_DEVICE int GetHeight(void) const
	{
		return m_Resolution[1];
	}

protected:
	Resolution2i	m_Resolution;
	T*				m_pData;
};

class CCudaRandomBuffer2D : public CCudaBuffer2D<unsigned int, false>
{
public:
	void Resize(const Resolution<int, 2>& Resolution)
	{
		CCudaBuffer2D::Resize(Resolution);

		unsigned int* pSeeds = (unsigned int*)malloc(GetSize());

		memset(pSeeds, 0, GetSize());

		for (int i = 0; i < GetNoElements(); i++)
			pSeeds[i] = rand();

		cudaMemcpy(m_pData, pSeeds, GetSize(), cudaMemcpyHostToDevice);

		free(pSeeds);
	}
};