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

#include "buffer.h"

namespace ExposureRender
{

template<class T>
class EXPOSURE_RENDER_DLL Buffer1D : public Buffer
{
public:
	HOST Buffer1D(const Enums::MemoryType& MemoryType = Enums::Host, const char* pName = "Buffer (1D)") :
		Buffer(MemoryType, pName),
		Resolution(0),
		NoElements(0),
		Data(NULL),
		Dirty(false)
	{
		DebugLog("%s: %s", __FUNCTION__, this->GetFullName());
	}

	HOST Buffer1D(const Buffer1D& Other) :
		Buffer(),
		Resolution(0),
		NoElements(0),
		Data(NULL),
		Dirty(false)
	{
		DebugLog("%s: Other = %s", __FUNCTION__, Other.GetFullName());
		
		*this = Other;
	}

	HOST virtual ~Buffer1D(void)
	{
		DebugLog(__FUNCTION__);
		this->Free();
	}

	HOST Buffer1D& operator = (const Buffer1D& Other)
	{
		DebugLog("%s: this = %s, Other = %s", __FUNCTION__, this->GetFullName(), Other.GetFullName());
		
		if (Other.Dirty)
		{
			this->Set(Other.MemoryType, Other.Resolution, Other.Data);
			Other.Dirty = false;
		}
		
		sprintf_s(this->Name, MAX_CHAR_SIZE, "Copy of %s", Other.Name);

		return *this;
	}

	HOST void Free(void)
	{
		DebugLog("%s: %s", __FUNCTION__, this->GetFullName());

		if (this->Data)
		{
			if (this->MemoryType == Enums::Host)
			{
				free(this->Data);
				this->Data = NULL;
			}

#ifdef __CUDA_ARCH__
			if (this->MemoryType == Enums::Device)
				Cuda::Free(this->Data);
#endif
		}
				
		this->Resolution	= Vec1i(0);
		this->NoElements	= 0;
		this->Dirty			= true;
	}

	HOST void Destroy(void)
	{
		DebugLog("%s: %s", __FUNCTION__, this->GetFullName());

		this->Resize(Vec1i(0));
		
		this->Dirty = true;
	}

	HOST void Reset(void)
	{
		DebugLog("%s: %s", __FUNCTION__, this->GetFullName());
		
		if (this->GetNoElements() <= 0)
			return;
		
		if (this->MemoryType == Enums::Host)
			memset(this->Data, 0, this->GetNoBytes());

#ifdef __CUDA_ARCH__
		if (this->MemoryType == Enums::Device)
			Cuda::MemSet(this->Data, 0, this->GetNoElements());
#endif
		
		this->Dirty = true;
	}

	HOST void Resize(const int& Resolution)
	{
		DebugLog("%s: %s, %d", __FUNCTION__, this->GetFullName(), Resolution);

		if (this->Resolution == Resolution)
			return;
		else
			this->Free();

		this->Resolution	= Resolution;
		this->NoElements	= this->Resolution;

		if (this->NoElements <= 0)
			return;

		if (this->MemoryType == Enums::Host)
			this->Data = (T*)malloc(this->GetNoBytes());

#ifdef __CUDA_ARCH__
		if (this->MemoryType == Enums::Device)
			Cuda::Allocate(this->Data, this->GetNoElements());
#endif

		this->Reset();
	}

	HOST void Set(const Enums::MemoryType& MemoryType, const Vec1i& Resolution, T* Data)
	{
		DebugLog("%s: %s, %d", __FUNCTION__, this->GetFullName(), Resolution);

		this->Resize(Resolution);

		if (this->NoElements <= 0)
			return;

		if (this->MemoryType == Enums::Host)
		{
			if (MemoryType == Enums::Host)
				memcpy(this->Data, Data, this->GetNoBytes());
			
#ifdef __CUDA_ARCH__
			if (MemoryType == Enums::Device)
				Cuda::MemCopyDeviceToHost(Data, this->Data, this->GetNoElements());
#endif
		}

#ifdef __CUDA_ARCH__
		if (this->MemoryType == Enums::Device)
		{
			if (MemoryType == Enums::Host)
				Cuda::MemCopyHostToDevice(Data, this->Data, this->GetNoElements());

			if (MemoryType == Enums::Device)
				Cuda::MemCopyDeviceToDevice(Data, this->Data, this->GetNoElements());
		}
#endif

		this->Dirty = true;
	}

	HOST_DEVICE int GetNoElements(void) const
	{
		return this->NoElements;
	}

	HOST_DEVICE int GetNoBytes(void) const
	{
		return this->GetNoElements() * sizeof(T);
	}

	HOST_DEVICE T& operator[](const int& i) const
	{
		return this->Data[i];
	}

	Enums::MemoryType	MemoryType;
	int					Resolution;
	int					NoElements;
	T*					Data;
	mutable bool		Dirty;
};

}
