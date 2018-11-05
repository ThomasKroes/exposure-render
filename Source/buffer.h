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

#include "geometry.h"
#include "wrapper.cuh"

namespace ExposureRender
{

template<class T>
class EXPOSURE_RENDER_DLL Buffer
{
public:
	HOST Buffer(const Enums::MemoryType& MemoryType = Enums::Host, const char* pName = "Untitled") :
		MemoryType(MemoryType),
		Data(NULL),
		NoElements(0),
		Dirty(false)
	{
		this->SetName(pName);
	}

	HOST Buffer(const Buffer& Other)
	{
		this->SetName(Other.GetName());
	}

	HOST const char* GetName() const
	{
		return this->Name;
	}

	HOST void SetName(const char* pName)
	{
		snprintf(this->Name, MAX_CHAR_SIZE, "%s", pName);
		this->UpdateFullName();
	}
	
	HOST const char* GetFullName() const
	{
		return this->FullName;
	}

	HOST void UpdateFullName()
	{
		char MemoryTypeName[MAX_CHAR_SIZE];

		switch (this->MemoryType)
		{
			case Enums::Host:
				snprintf(MemoryTypeName, MAX_CHAR_SIZE, "%s", "H");
				break;
			
			case Enums::Device:
				snprintf(MemoryTypeName, MAX_CHAR_SIZE, "%s", "D");
				break;

			default:
				snprintf(MemoryTypeName, MAX_CHAR_SIZE, "%s", "U");
				break;
		}

		snprintf(this->FullName, MAX_CHAR_SIZE, "['%s', %s]", this->Name, MemoryTypeName);
	}

	HOST_DEVICE virtual int GetNoBytes() const
	{
		return 0;
	}

	HOST virtual float GetMemorySize(const Enums::MemoryUnit& MemoryUnit) const
	{
		switch (MemoryUnit)
		{
			case Enums::KiloByte:
				return (float)this->GetNoBytes() / (1024.0f);

			case Enums::MegaByte:
				return (float)this->GetNoBytes() / (1024.0f * 1024.0f);
			
			case Enums::GigaByte:
				return (float)this->GetNoBytes() / (1024.0f * 1024.0f * 1024.0f);
		}

		return 0.0f;
	}

	HOST virtual void GetMemoryString(char* pMemoryString, const Enums::MemoryUnit& MemoryUnit = Enums::MegaByte) const
	{
		switch (MemoryUnit)
		{
			case Enums::KiloByte:	snprintf(pMemoryString, MAX_CHAR_SIZE, "%0.2f KB", this->GetMemorySize(Enums::KiloByte));		break;
			case Enums::MegaByte:	snprintf(pMemoryString, MAX_CHAR_SIZE, "%0.2f MB", this->GetMemorySize(Enums::MegaByte));		break;
			case Enums::GigaByte:	snprintf(pMemoryString, MAX_CHAR_SIZE, "%0.2f GB", this->GetMemorySize(Enums::GigaByte));		break;
		}
	}

	Enums::MemoryType	MemoryType;
	char				Name[MAX_CHAR_SIZE];
	char				FullName[MAX_CHAR_SIZE];
	T*					Data;
	int					NoElements;
	mutable bool		Dirty;
};

}
