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

namespace ExposureRender
{

namespace Cuda
{

template<typename D, typename H, int MaxSize = 32>
class List
{
public:
	HOST List(const char* pDeviceSymbol) :
		Map(),
		MapIt(),
		HashMap(),
		HashMapIt(),
		DeviceList(NULL),
		Counter(0),
		DeviceSymbol()
	{
		DebugLog(__FUNCTION__);
		snprintf(DeviceSymbol, MAX_CHAR_SIZE, "%s", pDeviceSymbol);
	}

	HOST ~List()
	{
		DebugLog(__FUNCTION__);
	}
	
	HOST bool Exists(const int& ID)
	{
		if (ID < 0)
			return false;

		this->MapIt = this->Map.find(ID);

		return this->MapIt != this->Map.end();
	}

	HOST void Bind(const H& Item)
	{
		DebugLog(__FUNCTION__);

		if (this->Map.size() + 1 >= MaxSize)
		{
			DebugLog("%s failed, max. no. items reached", __FUNCTION__);
			return;
		}
		
		const bool Exists = this->Exists(Item.ID);
		
		if (!Exists)
		{
			Item.ID = this->Counter;
			this->Map[Item.ID] = new D(Item);
			this->Counter++;
		}
		else
		{
			*(this->Map[Item.ID]) = Item;
		}

		this->Synchronize();
	}

	HOST void Unbind(const H& Item)
	{
		DebugLog(__FUNCTION__);

		if (!this->Exists(Item.ID))
		{
			DebugLog("%s failed, resource item with ID:%d does not exist", __FUNCTION__, Item.ID);
			return;
		}
		
		delete this->Map[Item.ID];
				
		this->MapIt = this->Map.find(Item.ID);
		this->Map.erase(this->MapIt);

		this->HashMapIt = this->HashMap.find(Item.ID);

		if (this->HashMapIt != this->HashMap.end())
			this->HashMap.erase(this->HashMapIt);

		this->Synchronize();
	}

	HOST void Synchronize(const int& ID = 0)
	{
//		DebugLog(__FUNCTION__);

		if (this->Map.size() <= 0)
			return; // DebugLog("%s failed, map is empty", __FUNCTION__);

		if (ID == 0)
		{
			D* pHostList = (D*)malloc(this->Map.size() * sizeof(D));
		
			int Size = 0;

			for (this->MapIt = this->Map.begin(); this->MapIt != this->Map.end(); this->MapIt++)
			{
				memcpy((void*)&pHostList[Size], (void*)this->MapIt->second, sizeof(D));
				HashMap[this->MapIt->first] = Size;
				Size++;
			}
			
			Cuda::Free(this->DeviceList);
			Cuda::Allocate(this->DeviceList, (int)this->Map.size());
			Cuda::MemCopyHostToDevice(pHostList, this->DeviceList, Size);
			Cuda::MemCopyHostToDeviceSymbol(&this->DeviceList, this->DeviceSymbol);
		
			free(pHostList);
		}
		else
		{
			if (!this->Exists(ID))
				return;

			Cuda::Free(this->DeviceList);
			Cuda::Allocate(this->DeviceList);
			
			this->MapIt = this->Map.find(ID);

			Cuda::MemCopyHostToDevice(this->MapIt->second, this->DeviceList);
			Cuda::MemCopyHostToDeviceSymbol(&this->DeviceList, this->DeviceSymbol);
		}
	}

	HOST D& operator[](const int& i)
	{
//		DebugLog(__FUNCTION__);

		if (!this->Exists(i))
		{
			char Message[MAX_CHAR_SIZE];

			snprintf(Message, MAX_CHAR_SIZE, "%s failed, resource item with ID:%d does not exist", __FUNCTION__, i);

			throw(Exception(Enums::Warning, Message));
		}

		return *this->Map[i];
	}

	map<int, D*>						Map;
	typename map<int, D*>::iterator		MapIt;
	map<int, int>						HashMap;
	typename map<int, int>::iterator	HashMapIt;
	D*									DeviceList;
	int									Counter;
	char								DeviceSymbol[MAX_CHAR_SIZE];
};

}

}
