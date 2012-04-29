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
#include "cuda.h"

namespace ExposureRender
{

template<class T>
class EXPOSURE_RENDER_DLL Buffer2D
{
public:
	HOST Buffer2D() :
		MemoryType(Enums::Host),
		Resolution(0),
		InvResolution(0.0f),
		Spacing(1.0f),
		InvSpacing(1.0f),
		Data(NULL),
		ModifiedTime(0)
	{
	}

	HOST Buffer2D(const Enums::MemoryType& MemoryType) :
		MemoryType(MemoryType),
		Resolution(0),
		InvResolution(0.0f),
		Spacing(1.0f),
		InvSpacing(1.0f),
		Data(NULL),
		ModifiedTime(0)
	{
	}

	HOST Buffer2D(const Enums::MemoryType& MemoryType, const Vec2i& Resolution, const Vec2f& Spacing, T* Data)
	{
		Buffer2D Other;

		Other.MemoryType	= MemoryType;
		Other.Resolution	= Resolution;
		Other.InvResolution	= 1.0f / Resolution;
		Other.Spacing		= Spacing;
		Other.InvSpacing	= 1.0f / Spacing;
		Other.Data			= Data;

		*this = Other;
	}

	HOST virtual ~Buffer2D(void)
	{
		this->Free();
	}

	HOST void Resize(const Vec2i& Resolution)
	{
		if (this->Resolution == Resolution)
			return;
		else
			this->Free();

		this->SetResolution(Resolution);

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
		if (this->GetNoElements() <= 0)
			printf("Buffer2D::Reset() failed: no elements in buffer!\n");
		
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
		else
			printf("Buffer2D::Free() failed: data pointer is NULL!");
		
		this->Resolution = Vec2i(0);

		this->ModifiedTime++;
	}

	HOST void Copy(const Buffer2D& Other)
	{
		if (this->ModifiedTime == Other.ModifiedTime)
			return;

		this->Resize(Resolution);

		if (this->MemoryType == Enums::Host)
		{
			if (Other.MemoryType == Enums::Host)
				memcpy(this->Data, Other.Data, this->GetNoBytes());
			
#ifdef __CUDA_ARCH__
			if (Other.MemoryType == Enums::Device)
				Cuda::MemCopyDeviceToHost(Other.Data, this->Data, this->GetNoElements());
#endif
		}

#ifdef __CUDA_ARCH__
		if (this->MemoryType == Enums::Device)
		{
			if (Other.MemoryType == Enums::Host)
				Cuda::MemCopyHostToDevice(Other.Data, this->Data, this->GetNoElements());

			if (Other.MemoryType == Enums::Device)
				Cuda::MemCopyDeviceToDevice(Other.Data, this->Data, this->GetNoElements());
		}
#endif

		this->ModifiedTime++;
	}

	HOST void SetSpacing(const Vec2f& Spacing)
	{
		this->Spacing		= Spacing;
		this->InvSpacing	= 1.0f / Spacing;
	}

	HOST void SetResolution(const Vec2i& Resolution)
	{
		this->Resolution	= Resolution;
		this->InvResolution	= 1.0f / Resolution;
	}

	HOST_DEVICE int GetNoElements(void) const
	{
		return this->Resolution[0] * this->Resolution[1];
	}

	HOST_DEVICE int GetNoBytes(void) const
	{
		return this->GetNoElements() * sizeof(T);
	}

	HOST_DEVICE T& operator()(const int& x = 0, const int& y = 0) const
	{
		return this->Data[y * this->Resolution[0] + x];
	}

	HOST_DEVICE T& operator()(const Vec2i& xy) const
	{
		return this->Data[xy[1] * this->Resolution[0] + xy[0]];
	}

	HOST_DEVICE T& operator[](const int& i) const
	{
		return this->Data[i];
	}

	HOST Buffer2D& operator = (const Buffer2D& Other)
	{
		this->Copy(Other);
		 
		return *this;
	}

	Enums::MemoryType	MemoryType;
	Vec2i				Resolution;
	Vec2f				InvResolution;
	Vec2f				Spacing;
	Vec2f				InvSpacing;
	T*					Data;
	long				ModifiedTime;
};

template<class T>
class EXPOSURE_RENDER_DLL Buffer3D
{
public:
	HOST Buffer3D() :
		MemoryType(Enums::Host),
		Resolution(0),
		InvResolution(0.0f),
		Spacing(1.0f),
		InvSpacing(1.0f),
		Data(NULL),
		ModifiedTime(0)
	{
	}

	HOST Buffer3D(const Enums::MemoryType& MemoryType) :
		MemoryType(MemoryType),
		Resolution(0),
		InvResolution(0.0f),
		Spacing(1.0f),
		InvSpacing(1.0f),
		Data(NULL),
		ModifiedTime(0)
	{
	}

	HOST Buffer3D(const Enums::MemoryType& MemoryType, const Vec3i& Resolution, const Vec3f& Spacing, T* Data)
	{
		Buffer3D Other;

		Other.MemoryType	= MemoryType;
		Other.Resolution	= Resolution;
		Other.InvResolution	= 1.0f / Resolution;
		Other.Spacing		= Spacing;
		Other.InvSpacing	= 1.0f / Spacing;
		Other.Data			= Data;

		*this = Other;
	}

	HOST virtual ~Buffer3D(void)
	{
		this->Free();
	}

	HOST void Resize(const Vec3i& Resolution)
	{
		if (this->Resolution == Resolution)
			return;
		else
			this->Free();

		this->SetResolution(Resolution);

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
		if (this->GetNoElements() <= 0)
			printf("Buffer3D::Reset() failed: no elements in buffer!");
		
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
		else
			printf("Buffer3D::Free() failed: data pointer is NULL!");
		
		this->Resolution = Vec3i(0);

		this->ModifiedTime++;
	}

	HOST void Copy(const Buffer3D& Other)
	{
		if (this->ModifiedTime == Other.ModifiedTime)
			return;

		this->Resize(Resolution);

		if (this->MemoryType == Enums::Host)
		{
			if (Other.MemoryType == Enums::Host)
				memcpy(this->Data, Other.Data, this->GetNoBytes());
			
#ifdef __CUDA_ARCH__
			if (Other.MemoryType == Enums::Device)
				Cuda::MemCopyDeviceToHost(Other.Data, this->Data, this->GetNoElements());
#endif
		}

#ifdef __CUDA_ARCH__
		if (this->MemoryType == Enums::Device)
		{
			if (Other.MemoryType == Enums::Host)
				Cuda::MemCopyHostToDevice(Other.Data, this->Data, this->GetNoElements());

			if (Other.MemoryType == Enums::Device)
				Cuda::MemCopyDeviceToDevice(Other.Data, this->Data, this->GetNoElements());
		}
#endif

		this->ModifiedTime++;
	}

	HOST void SetSpacing(const Vec3f& Spacing)
	{
		this->Spacing		= Spacing;
		this->InvSpacing	= 1.0f / Spacing;
	}

	HOST void SetResolution(const Vec3i& Resolution)
	{
		this->Resolution	= Resolution;
		this->InvResolution	= 1.0f / Resolution;
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
		this->Copy(Other);
		 
		return *this;
	}

	Enums::MemoryType	MemoryType;
	Vec3i				Resolution;
	Vec3f				InvResolution;
	Vec3f				Spacing;
	Vec3f				InvSpacing;
	T*					Data;
	long				ModifiedTime;
};

class RandomSeedBuffer2D : public Buffer2D<unsigned int>
{
public:
	HOST RandomSeedBuffer2D(const Enums::MemoryType& MemoryType) :
		Buffer2D(MemoryType)
	{
	}

	void Resize(const Vec2i& Resolution)
	{
		Buffer2D Seeds(Enums::Host);

		Seeds.Resize(Resolution);

		for (int i = 0; i < Seeds.GetNoElements(); i++)
			Seeds[i] = rand();

		this->Copy(Seeds);
	}
};

}
