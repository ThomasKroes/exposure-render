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
class EXPOSURE_RENDER_DLL Buffer2D : public Buffer<T>
{
public:
	HOST Buffer2D(const Enums::MemoryType& MemoryType = Enums::Host, const char* pName = "Buffer (2D)") :
		Buffer<T>(MemoryType, pName),
		Resolution(0)
	{
		DebugLog("%s: %s", __FUNCTION__, this->GetFullName());
	}

	HOST Buffer2D(const Buffer2D& Other) :
		Buffer<T>(),
		Resolution(0)
	{
		DebugLog("%s: Other = %s", __FUNCTION__, Other.GetFullName());
		
		*this = Other;
	}

	HOST virtual ~Buffer2D(void)
	{
		DebugLog(__FUNCTION__);
		this->Free();
	}

	HOST Buffer2D& operator = (const Buffer2D& Other)
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

		char MemoryString[MAX_CHAR_SIZE];
		
		this->GetMemoryString(MemoryString, Enums::MegaByte);

		if (this->Data)
		{
			if (this->MemoryType == Enums::Host)
			{
				free(this->Data);
				this->Data = NULL;
				DebugLog("Freed %s on host", MemoryString);
			}

#ifdef __CUDA_ARCH__
			if (this->MemoryType == Enums::Device)
			{
				Cuda::Free(this->Data);
				DebugLog("Freed %s on device", MemoryString);
			}
#endif
		}
				
		this->Resolution	= Vec2i(0);
		this->NoElements	= 0;
		this->Dirty			= true;
	}

	HOST void Destroy(void)
	{
		DebugLog("%s: %s", __FUNCTION__, this->GetFullName());

		this->Resize(Vec2i(0));
		
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

	HOST void Resize(const Vec2i& Resolution)
	{
		DebugLog("%s", __FUNCTION__);

		if (this->Resolution == Resolution)
			return;
		else
			this->Free();

		this->Resolution = Resolution;

		DebugLog("Resolution = [%d x %d]", this->Resolution[0], this->Resolution[1]);

		this->NoElements = this->Resolution[0] * this->Resolution[1];

		if (this->NoElements <= 0)
			return;
		
		DebugLog("No. Elements = %d", this->NoElements);
		
		char MemoryString[MAX_CHAR_SIZE];
		
		this->GetMemoryString(MemoryString, Enums::MegaByte);

		if (this->MemoryType == Enums::Host)
		{
			this->Data = (T*)malloc(this->GetNoBytes());
			DebugLog("Allocated %s on host", MemoryString);
		}

#ifdef __CUDA_ARCH__
		if (this->MemoryType == Enums::Device)
		{
			Cuda::Allocate(this->Data, this->GetNoElements());
			DebugLog("Allocated %s on device", MemoryString);
		}
#endif

		this->Reset();
	}

	HOST void Set(const Enums::MemoryType& MemoryType, const Vec2i& Resolution, T* Data)
	{
		DebugLog("%s: %s, %d x %d", __FUNCTION__, this->GetFullName(), Resolution[0], Resolution[1]);

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

	HOST_DEVICE virtual int GetNoBytes(void) const
	{
		return this->GetNoElements() * sizeof(T);
	}

	HOST_DEVICE T* GetData(void) const
	{
		return this->Data;
	}

	HOST_DEVICE T& operator()(const int& X = 0, const int& Y = 0) const
	{
		const Vec2i ClampedXY(Clamp(X, 0, this->Resolution[0] - 1), Clamp(Y, 0, this->Resolution[1] - 1));
		return this->Data[ClampedXY[1] * this->Resolution[0] + ClampedXY[0]];
	}

	HOST_DEVICE T& operator()(const Vec2i& XY) const
	{
		const Vec2i ClampedXY(Clamp(XY[0], 0, this->Resolution[0] - 1), Clamp(XY[1], 0, this->Resolution[1] - 1));
		return this->Data[ClampedXY[1] * this->Resolution[0] + ClampedXY[0]];
	}

	HOST_DEVICE T operator()(const Vec2f& XY, const bool Normalized = false) const
	{
		const Vec2f UV = Normalized ? XY * Vec2f(this->Resolution[0], this->Resolution[1]) : XY;

		int Coord[2][2] =
		{
			{ floorf(UV[0]), ceilf(UV[0]) },
			{ floorf(UV[1]), ceilf(UV[1]) },
		};

		const float du = UV[0] - Coord[0][0];
		const float dv = UV[1] - Coord[1][0];

		Coord[0][0] = min(max(Coord[0][0], 0), this->Resolution[0] - 1);
		Coord[0][1] = min(max(Coord[0][1], 0), this->Resolution[0] - 1);
		Coord[1][0] = min(max(Coord[1][0], 0), this->Resolution[1] - 1);
		Coord[1][1] = min(max(Coord[1][1], 0), this->Resolution[1] - 1);

		T Values[4] = 
		{
			T((*this)(Coord[0][0], Coord[1][0])),
			T((*this)(Coord[0][1], Coord[1][0])),
			T((*this)(Coord[0][0], Coord[1][1])),
			T((*this)(Coord[0][1], Coord[1][1]))
		};

		return (1.0f - dv) * ((1.0f - du) * Values[0] + du * Values[1]) + dv * ((1.0f - du) * Values[2] + du * Values[3]);
	}

	HOST_DEVICE T& operator[](const int& ID) const
	{
		const int ClampedID = Clamp(ID, 0, this->NoElements - 1);
		return this->Data[ClampedID];
	}

	Vec2i	Resolution;
};

class RandomSeedBuffer2D : public Buffer2D<unsigned int>
{
public:
	HOST RandomSeedBuffer2D(const Enums::MemoryType& MemoryType, const char* pName) :
		Buffer2D(MemoryType, pName)
	{
	}

	void Resize(const Vec2i& Resolution)
	{
		const int NoSeeds = Resolution[0] * Resolution[1];

		unsigned int* pSeeds = new unsigned int[NoSeeds];

		for (int i = 0; i < NoSeeds; i++)
			pSeeds[i] = rand();

		this->Set(Enums::Host, Resolution, pSeeds);

		delete[] pSeeds;
	}
};

}
