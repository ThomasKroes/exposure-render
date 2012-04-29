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
class EXPOSURE_RENDER_DLL Buffer3D : public Buffer
{
public:
	HOST Buffer3D(const Enums::MemoryType& MemoryType = Enums::Host, const char* pName = "Buffer (3D)") :
		Buffer(MemoryType, pName),
		Resolution(0),
		Data(NULL),
		ModifiedTime(0)
	{
		DebugLog("Creating 3D Buffer: %s", this->GetFullName());
	}

	HOST virtual ~Buffer3D(void)
	{
		this->Free();
	}

	HOST void Resize(const Vec3i& Resolution)
	{
		DebugLog("Resizing 3D buffer: %s, %d x %d x %d", this->GetFullName(), Resolution[0], Resolution[1], Resolution[2]);

		if (this->Resolution == Resolution)
			return;
		else
			this->Free();

		this->Resolution = Resolution;

		if (this->GetNoElements() <= 0)
			return;

		if (this->MemoryType == Enums::Host)
			this->Data = (T*)malloc(this->GetNoBytes());

#ifdef __CUDA_ARCH__
		if (this->MemoryType == Enums::Device)
			Cuda::Allocate(this->Data, this->GetNoElements());
#endif

		this->Reset();
	}

	HOST void Reset(void)
	{
		DebugLog("Resetting 3D buffer: %s", this->GetFullName());

		if (this->GetNoElements() <= 0)
			return;
		
		if (this->MemoryType == Enums::Host)
			memset(this->Data, 0, this->GetNoBytes());

#ifdef __CUDA_ARCH__
		if (this->MemoryType == Enums::Device)
			Cuda::MemSet(this->Data, 0, this->GetNoElements());
#endif

		this->ModifiedTime++;
	}

	HOST void Free(void)
	{
		DebugLog("Freeing 3D buffer: %s", this->GetFullName());

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
				
		this->Resolution = Vec3i(0);

		this->ModifiedTime++;
	}

	HOST void Set(const Enums::MemoryType& MemoryType, const Vec3i& Resolution, T* Data)
	{
		DebugLog("Setting 3D buffer: %s, %d x %d x %d", this->GetFullName(), Resolution[0], Resolution[1], Resolution[2]);

		this->Resize(Resolution);

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

		this->ModifiedTime++;
	}

	HOST_DEVICE int GetNoElements(void) const
	{
		return this->Resolution[0] * this->Resolution[1] * this->Resolution[2];
	}

	HOST_DEVICE int GetNoBytes(void) const
	{
		return this->GetNoElements() * sizeof(T);
	}

	HOST_DEVICE T& operator()(const int& x = 0, const int& y = 0, const int& z = 0) const
	{
		return this->Data[z * this->Resolution[0] * this->Resolution[1] + y * this->Resolution[0] + x];
	}

	HOST_DEVICE T& operator()(const Vec3i& xyz) const
	{
		return this->Data[xyz[2] * this->Resolution[0] * this->Resolution[1] + xyz[1] * this->Resolution[0] + xyz[0]];
	}

	HOST_DEVICE T& operator[](const int& i) const
	{
		return this->Data[i];
	}

	HOST Buffer3D& operator = (const Buffer3D& Other)
	{
		DebugLog("Assigning %s to %s", Other.GetFullName(), this->GetFullName());

		this->Set(Other.MemoryType, Other.Resolution, Other.Data);
		
		sprintf_s(this->Name, MAX_CHAR_SIZE, "Copy of %s", Other.Name);

		return *this;
	}

	Enums::MemoryType	MemoryType;
	Vec3i				Resolution;
	T*					Data;
	long				ModifiedTime;
};

}
