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

	virtual ~CCudaBuffer2D(void)
	{
		Free();
	}

	CCudaBuffer2D::CCudaBuffer2D(const CCudaBuffer2D& Other)
	{
		*this = Other;
	}

	CCudaBuffer2D& CCudaBuffer2D::operator=(const CCudaBuffer2D& Other)
	{
		m_Resolution	= Other.m_Resolution;
		m_pData			= Other.m_pData;
		m_Pitch			= Other.m_Pitch;

		return *this;
	}

	void Resize(const CResolution2D& Resolution)
	{
		m_Resolution = Resolution;

		Free();

		if (Pitched)
			HandleCudaError(cudaMallocPitch(&m_pData, &m_Pitch, GetWidth(), GetHeight()));
		else
			HandleCudaError(cudaMalloc(&m_pData, GetSize()));
	}

	void Reset(void)
	{
		HandleCudaError(cudaMemset(m_pData, 0, GetSize()));
	}

	void Free(void)
	{
		HandleCudaError(cudaFree(m_pData));
		m_pData = NULL;
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

	HOD T Get(const int& X, const int& Y)
	{
		if (Pitched)
			return m_pData[Y * (m_Pitch / sizeof(T)) + X];
		else
			return m_pData[Y * GetWidth() + X];
	}

	HOD T& GetRef(const int& X, const int& Y)
	{
		if (Pitched)
			return m_pData[Y * m_Pitch / sizeof(T) + X];
		else
			return m_pData[Y * GetWidth() + X];
	}

	HOD T* GetPtr(const int& X, const int& Y)
	{
		if (Pitched)
			return &m_pData[Y * m_Pitch / sizeof(T) + X];
		else
			return &m_pData[Y * GetWidth() + X];
	}

	HOD void Set(T& Value, const int& X, const int& Y)
	{
		if (Pitched)
			m_pData[Y * m_Pitch + X] = Value;
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