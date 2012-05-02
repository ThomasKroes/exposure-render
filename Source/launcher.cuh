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

#include "montecarlo.h"

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
		sprintf_s(DeviceSymbol, MAX_CHAR_SIZE, "%s", pDeviceSymbol);
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

	HOST void Synchronize(const int& Offset = 0)
	{
		DebugLog(__FUNCTION__);

		if (this->Map.size() <= 0)
			DebugLog("%s failed, map is empty", __FUNCTION__);

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
		Cuda::MemCopyHostToDeviceSymbol(&this->DeviceList, this->DeviceSymbol, 1, Offset);

		free(pHostList);
	}

	HOST D* operator[](const int& i)
	{
		DebugLog(__FUNCTION__);

		if (!this->Exists(i))
		{
			DebugLog("%s failed, resource item with ID:%d does not exist", __FUNCTION__, i);
			return NULL;
		}

		return this->Map[i];
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
