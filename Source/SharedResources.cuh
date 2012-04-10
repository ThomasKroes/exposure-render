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

	ResourceList()
	{
		this->Count = 0;
	}

	HOST_DEVICE T& operator[](const int& i)
	{
		return this->List[i];
	}

private:
	T List[MaxSize];
};

template<class T, int MaxSize>
struct SharedResources
{
	map<int, T>						Resources;
	int								Counter;
	
	ResourceList<T, MaxSize>*		DevicePtr;
	ResourceList<T, MaxSize>*		DeviceAllocation;

	typename map<int, T>::iterator	It;

	SharedResources(ResourceList<T, MaxSize>* DevicePtr)
	{
		this->Counter			= 0;
		this->DevicePtr			= DevicePtr;
		this->DeviceAllocation	= NULL;
	}

	~SharedResources()
	{
//		CUDA::Free(this->DeviceAllocation);
	}
	
	bool Exists(int ID)
	{
		if (ID < 0)
			return false;

		It = Resources.find(ID);

		return It != Resources.end();
	}

	void Bind(T Resource, int& ID)
	{
		this->Resources[ID] = Resource;

		if (this->Exists(ID))
		{
			ID = Counter;
			Counter++;
		}

		this->Synchronize();
	}

	void Unbind(int ID)
	{
		It = this->Resources.find(ID);

		if (It != Resources.end())
			Resources.erase(It);

		this->Synchronize();
	}

	void Synchronize()
	{
		if (Resources.empty())
			return;

		ResourceList<T, MaxSize> ResourceList;

		for (It = Resources.begin(); It != Resources.end(); It++)
		{
			ResourceList[ResourceList.Count] = It->second;
			ResourceList.Count++;
		}

		if (this->DeviceAllocation == NULL)
			cudaMalloc(&DeviceAllocation, sizeof(ResourceList));

		cudaMemcpy(DeviceAllocation, &ResourceList, sizeof(ResourceList), cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(DevicePtr, &DeviceAllocation, sizeof(DevicePtr));
	}
};

}
