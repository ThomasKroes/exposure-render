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
#include "CudaUtilities.h"

namespace ExposureRender
{

template<class T, bool Pitched>
class CCudaBuffer2D
{
public:
	CCudaBuffer2D(void) :
		Resolution(),
		pData(NULL),
		Pitch(0)
	{
	}

	HOST void Resize(Resolution2i Resolution)
	{
		if (this->Resolution != Resolution)
			this->Free();

		this->Resolution = Resolution;

		if (this->GetNoElements() <= 0)
			return;

		if (Pitched)
			Cuda::AllocatePiched(this->pData, this->Pitch, this->GetWidth(), this->GetHeight());
		else
			Cuda::Allocate(this->pData, this->GetNoElements());

		this->Reset();
	}

	HOST void Reset(void)
	{
		if (this->GetSize() <= 0)
			return;

		Cuda::MemSet(this->pData, 0, this->GetNoElements());
	}

	HOST void Free(void)
	{
		if (this->pData)
		{
			Cuda::Free(this->pData);
		}
		
		this->Pitch			= 0;
		this->Resolution	= Resolution2i();
	}

	HOST_DEVICE T& operator()(const int X, const int Y)
	{
		if (Pitched)
			return this->pData[Y * (this->GetPitch() / sizeof(T)) + X];
		else
			return this->pData[Y * this->GetWidth() + X];
	}

	HOST_DEVICE int GetNoElements(void) const
	{
		return this->Resolution.GetNoElements();
	}

	HOST_DEVICE int GetSize(void) const
	{
		if (Pitched)
			return this->Resolution[1] * this->Pitch;
		else
			return this->GetNoElements() * sizeof(T);
	}

	HOST_DEVICE T Get(const int& X = 0, const int& Y = 0)
	{
		if (X > this->GetWidth() || Y > this->GetHeight())
			return T();

		if (Pitched)
			return this->pData[Y * (this->GetPitch() / sizeof(T)) + X];
		else
			return this->pData[Y * this->GetWidth() + X];
	}

	HOST_DEVICE T& GetRef(const int& X = 0, const int& Y = 0)
	{
		if (X > this->GetWidth() || Y > this->GetHeight())
			return T();

		if (Pitched)
			return this->pData[Y * (this->GetPitch() / sizeof(T)) + X];
		else
			return this->pData[Y * this->GetWidth() + X];
	}

	HOST_DEVICE T* GetPtr(const int& X = 0, const int& Y = 0)
	{
		if (X > this->GetWidth() || Y > this->GetHeight())
			return NULL;

		if (Pitched)
			return &this->pData[Y * (this->GetPitch() / sizeof(T)) + X];
		else
			return &this->pData[Y * this->GetWidth() + X];
	}

	HOST_DEVICE void Set(T Value, int X = 0, int Y = 0)
	{
		if (X > this->GetWidth() || Y > this->GetHeight())
			return;

		if (Pitched)
			this->pData[Y * (this->GetPitch() / sizeof(T)) + X] = Value;
		else
			this->pData[Y * this->GetWidth() + X] = Value;
	}

	HOST_DEVICE int GetWidth(void) const
	{
		return this->Resolution[0];
	}

	HOST_DEVICE int GetHeight(void) const
	{
		return this->Resolution[1];
	}

	HOST_DEVICE int GetPitch(void) const
	{
		if (Pitched)
			return this->Pitch;
		else
			return this->GetWidth() * sizeof(T);
	}

protected:
	Resolution2i		Resolution;
	T*					pData;
	size_t				Pitch;
};

template<class T>
class CHostBuffer2D
{
public:
	CHostBuffer2D(void) :
		Resolution(),
		pData(NULL)
	{
	}

	virtual ~CHostBuffer2D(void)
	{
		this->Free();
	}

	void Resize(Resolution2i Resolution)
	{
		if (this->Resolution == Resolution)
			return;

		if (this->Resolution != Resolution)
			this->Free();

		this->Resolution = Resolution;

		if (this->GetNoElements() <= 0)
			return;

		this->pData = (T*)malloc(this->GetSize());

		this->Reset();
	}

	void Reset(void)
	{
		if (this->GetSize() <= 0)
			return;

		memset(this->pData, 0, this->GetSize());
	}

	void Free(void)
	{
		if (this->pData)
		{
			free(this->pData);
			this->pData = NULL;
		}
		
		this->Resolution = Resolution2i();
	}

	HOST_DEVICE int GetNoElements(void) const
	{
		return this->Resolution.GetNoElements();
	}

	HOST_DEVICE int GetSize(void) const
	{
		return this->GetNoElements() * sizeof(T);
	}

	HOST_DEVICE T Get(const int& X = 0, const int& Y = 0)
	{
		if (X > this->GetWidth() || Y > this->GetHeight())
			return T();

		return this->pData[Y * this->GetWidth() + X];
	}

	HOST_DEVICE T& GetRef(const int& X = 0, const int& Y = 0)
	{
		if (X > this->GetWidth() || Y > this->GetHeight())
			return T();

		return this->pData[Y * this->GetWidth() + X];
	}

	HOST_DEVICE T* GetPtr(const int& X = 0, const int& Y = 0)
	{
		if (X > this->GetWidth() || Y > this->GetHeight())
			return NULL;

		return &this->pData[Y * this->GetWidth() + X];
	}

	HOST_DEVICE void Set(T& Value, const int& X = 0, const int& Y = 0)
	{
		if (X > this->GetWidth() || Y > this->GetHeight())
			return;

		this->pData[Y * this->GetWidth() + X] = Value;
	}

	HOST_DEVICE int GetWidth(void) const
	{
		return this->Resolution[0];
	}

	HOST_DEVICE int GetHeight(void) const
	{
		return this->Resolution[1];
	}

protected:
	Resolution2i	Resolution;
	T*				pData;
};

class CCudaRandomBuffer2D : public CCudaBuffer2D<unsigned int, false>
{
public:
	void Resize(Resolution2i Resolution)
	{
		CCudaBuffer2D::Resize(Resolution);

		unsigned int* pSeeds = (unsigned int*)malloc(this->GetSize());

		memset(pSeeds, 0, this->GetSize());

		for (int i = 0; i < this->GetNoElements(); i++)
			pSeeds[i] = rand();

		cudaMemcpy(this->pData, pSeeds, this->GetSize(), cudaMemcpyHostToDevice);

		free(pSeeds);
	}
};

}