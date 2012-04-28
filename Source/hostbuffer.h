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

template<class T>
class Buffer2D
{
public:
	HOST Buffer2D(const Enums::MemoryType& MemoryType) :
		MemoryType(MemoryType)
		Resolution(),
		Data(NULL)
	{
	}

	HOST virtual ~Buffer2D(void)
	{
		this->Free();
	}

	HOST void Resize(const Resolution2i& Resolution)
	{
		if (this->Resolution == Resolution)
			return;
		else
			this->Free();

		this->Resolution = Resolution;

		if (this->GetNoElements() <= 0)
			return;

		if (this->MemoryType == Host)
			this->Data = (T*)malloc(this->GetNoBytes());

		if (this->MemoryType == Device)
			Cuda::Allocate(this->Data, this->GetNoElements());

		this->Reset();
	}

	HOST void Reset(void)
	{
		if (this->GetSize() <= 0)
			throw(Exception(Enums::Warning, "Buffer2D::Reset() failed: no elements in buffer!"));
		
		if (this->MemoryType == Host)
			memset(this->Data, 0, this->GetNoBytes());

		if (this->MemoryType == Device)
			Cuda::MemSet(this->Data, 0, this->GetNoElements());
	}

	HOST void Free(void)
	{
		if (this->Data)
		{
			if (this->MemoryType == Host)
			{
				free(this->Data);
				this->Data = NULL;
			}

			if (this->MemoryType == Device)
				Cuda::Free(this->Data);
		}
		else
			throw(Exception(Enums::Warning, "Buffer2D::Free() failed: data pointer is NULL!"));
		
		this->Resolution = Resolution2i(0, 0);
	}

	HOST_DEVICE int GetNoElements(void) const
	{
		return this->Resolution.GetNoElements();
	}

	HOST_DEVICE int GetNoBytes(void) const
	{
		return this->GetNoElements() * sizeof(T);
	}

	HOST_DEVICE T& operator()(const int& X = 0, const int& Y = 0) const
	{
		return this->Data[Y * this->Resolution[0] + X];
	}

	HOST_DEVICE T& operator()(const Vec2i& XY) const
	{
		return this->Data[XY[1] * this->Resolution[0] + XY[0]];
	}

	HOST Buffer2D& operator = (const Buffer2D& Other)
	{
		this->Resolution = Other.Resolution;

		this->Resize(Other.Resolution);

		if (this->MemoryType == Host)
		{
			if (Other.MemoryType == Host)
				memcpy(this->Data, Other.Data);
			
			if (Other.MemoryType == Device)
				Cuda::MemCopyDeviceToHost(Other.Data, this->Data, this->GetNoElements());
		}

		if (this->MemoryType == Device)
		{
			if (Other.MemoryType == Host)
				Cuda::MemCopyHostToDevice(Other.Data, this->Data, this->GetNoElements());

			if (Other.MemoryType == Device)
				Cuda::MemCopyDeviceToDevice(Other.Data, this->Data, this->GetNoElements());
		}

		return *this;
	}

protected:
	Enums::MemoryType	MemoryType;
	Resolution2i		Resolution;
	T*					Data;
};

}
