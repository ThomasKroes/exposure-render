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

#include "Defines.cuh"
#include "General.cuh"
#include "CudaUtilities.cuh"

#include <map>

using namespace std;

namespace ExposureRender
{

template<class T, int MaxSize>
struct ResourceList
{
	int		Count;

	HOST ResourceList()
	{
		this->Count = 0;
	}

	HOST_DEVICE T& Get(const int& ID)
	{
		return this->List[ID];
	}

	HOST void Add(const T& Resource)
	{
		if (this->Count + 1 >= MaxSize)
			return;

		this->List[this->Count] = Resource;
		this->Count++;
	}

	HOST void Reset()
	{
		this->Count = 0;
	}

private:
	T List[MaxSize];
};

template<typename T, int MaxSize>
struct SharedResources
{
	typename map<int, T>			Resources;
	int								Counter;
	char							DeviceSymbol[MAX_CHAR_SIZE];
	ResourceList<T, MaxSize>*		DeviceAllocation;
	ResourceList<T, MaxSize>		List;
	typename map<int, T>::iterator	It;

	HOST SharedResources(const char* pDeviceSymbol)
	{
		this->Counter = 0;

		sprintf_s(DeviceSymbol, MAX_CHAR_SIZE, "%s", pDeviceSymbol);

		this->DeviceAllocation = NULL;
	}

	HOST ~SharedResources()
	{
//		CUDA::Free(this->DeviceAllocation);
	}
	
	HOST bool Exists(int ID)
	{
		if (ID < 0)
			return false;

		It = Resources.find(ID);

		return It != Resources.end();
	}

	HOST void Bind(const T& Resource, int& ID)
	{
		if (this->Resources.size() >= MaxSize)
			throw(ErException(Enums::Warning, "Maximum number of resources reached"));

		const bool Exists = this->Exists(ID);

		this->Resources[Counter] = Resource;

		if (!Exists)
		{
			ID = Counter;
			Counter++;
		}

		this->Synchronize();
	}

	HOST void Unbind(int ID)
	{
		if (!this->Exists(ID))
			return;

		It = this->Resources.find(ID);

		if (It != Resources.end())
			Resources.erase(It);

		this->Synchronize();
	}

	HOST void Synchronize()
	{
		if (Resources.empty())
			return;

		this->List.Reset();

		for (It = Resources.begin(); It != Resources.end(); It++)
			this->List.Add(It->second);
		
		if (this->DeviceAllocation == NULL)
			cudaMalloc(&this->DeviceAllocation, sizeof(this->List));	

		cudaMemcpy(DeviceAllocation, &this->List, sizeof(this->List), cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(DeviceSymbol, &DeviceAllocation, sizeof(&this->List));
	}

	HOST T& operator[](const int& i)
	{
		It = this->Resources.find(i);

		if (It == Resources.end())
			throw(ErException(Enums::Fatal, "Resource does not exist"));

		return this->Resources[i];
	}
};

}
