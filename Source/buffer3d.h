/*
    Exposure Render: An interactive photo-realistic volume rendering framework
    Copyright (C) 2011 Thomas Kroes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "buffer.h"

namespace ExposureRender
{

template<class T>
class EXPOSURE_RENDER_DLL Buffer3D : public Buffer<T>
{
public:
	HOST Buffer3D(const Enums::MemoryType& MemoryType = Enums::Host, const char* pName = "Buffer (3D)") :
		Buffer<T>(MemoryType, pName),
		Resolution(0)
	{
		DebugLog("%s: %s", __FUNCTION__, this->GetFullName());
	}

	HOST Buffer3D(const Buffer3D& Other) :
		Buffer<T>(),
		Resolution(0)
	{
		DebugLog("%s: Other = %s", __FUNCTION__, Other.GetFullName());
		
		*this = Other;
	}

	HOST virtual ~Buffer3D(void)
	{
		DebugLog(__FUNCTION__);
		this->Free();
	}

	HOST Buffer3D& operator = (const Buffer3D& Other)
	{
		DebugLog("%s: this = %s, Other = %s", __FUNCTION__, this->GetFullName(), Other.GetFullName());
		
		if (Other.Dirty)
		{
			this->Set(Other.MemoryType, Other.Resolution, Other.Data);
			Other.Dirty = false;
		}
		
		snprintf(this->Name, MAX_CHAR_SIZE, "Copy of %s", Other.Name);

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
				
		this->Resolution	= Vec3i(0);
		this->NoElements	= 0;
		this->Dirty			= true;
	}

	HOST void Destroy(void)
	{
		DebugLog("%s: %s", __FUNCTION__, this->GetFullName());

		this->Resize(Vec3i(0));
		
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

	HOST void Resize(const Vec3i& Resolution)
	{
		DebugLog("%s", __FUNCTION__);
		
		if (this->Resolution == Resolution)
			return;
		else
			this->Free();
		
		this->Resolution = Resolution;
		
		DebugLog("Resolution = [%d x %d x %d]", this->Resolution[0], this->Resolution[1], this->Resolution[2]);

		this->NoElements = this->Resolution[0] * this->Resolution[1] * this->Resolution[2];
		
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

	HOST void Set(const Enums::MemoryType& MemoryType, const Vec3i& Resolution, T* Data)
	{
		DebugLog("%s: %s, %d x %d x %d", __FUNCTION__, this->GetFullName(), Resolution[0], Resolution[1], Resolution[2]);

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

	HOST_DEVICE T& operator()(const int& X = 0, const int& Y = 0, const int& Z = 0) const
	{
		const Vec3i ClampedXYZ(Clamp(X, 0, this->Resolution[0] - 1), Clamp(Y, 0, this->Resolution[1] - 1), Clamp(Z, 0, this->Resolution[2] - 1));
		return this->Data[ClampedXYZ[2] * this->Resolution[0] * this->Resolution[1] + ClampedXYZ[1] * this->Resolution[0] + ClampedXYZ[0]];
	}

	HOST_DEVICE T& operator()(const Vec3i& XYZ) const
	{
		const Vec3i ClampedXYZ(Clamp(XYZ[0], 0, this->Resolution[0] - 1), Clamp(XYZ[1], 0, this->Resolution[1] - 1), Clamp(XYZ[2], 0, this->Resolution[2] - 1));
		return this->Data[ClampedXYZ[2] * this->Resolution[0] * this->Resolution[1] + ClampedXYZ[1] * this->Resolution[0] + ClampedXYZ[0]];
	}
	
	HOST_DEVICE T operator()(const Vec3f& XYZ, const bool Normalized = false) const
	{
		const Vec3f UVW = Normalized ? XYZ * Vec3f((float)this->Resolution[0], (float)this->Resolution[1], (float)this->Resolution[2]) : XYZ;

		const int vx = (int)floorf(UVW[0]);
		const int vy = (int)floorf(UVW[1]);
		const int vz = (int)floorf(UVW[2]);

		const float dx = UVW[0] - vx;
		const float dy = UVW[1] - vy;
		const float dz = UVW[2] - vz;

		const T d00 = Lerp(dx, (*this)(vx, vy, vz), (*this)(vx+1, vy, vz));
		const T d10 = Lerp(dx, (*this)(vx, vy+1, vz), (*this)(vx+1, vy+1, vz));
		const T d01 = Lerp(dx, (*this)(vx, vy, vz+1), (*this)(vx+1, vy, vz+1));
		const T d11 = Lerp(dx, (*this)(vx, vy+1, vz+1), (*this)(vx+1, vy+1, vz+1));
		const T d0	= Lerp(dy, d00, d10);
		const T d1 	= Lerp(dy, d01, d11);

		return Lerp(dz, d0, d1);
	}

	HOST_DEVICE T& operator[](const int& ID) const
	{
		const int ClampedID = Clamp(ID, 0, this->NoElements - 1);
		return this->Data[ClampedID];
	}

	Vec3i	Resolution;
};

}
