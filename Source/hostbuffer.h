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

#include "geometry.h"

namespace ExposureRender
{

template<class T, bool pitched = false>
class HostBuffer2D
{
public:
	HostBuffer2D(void) :
		Resolution(),
		pData(NULL)
	{
	}

	virtual ~HostBuffer2D(void)
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

}
