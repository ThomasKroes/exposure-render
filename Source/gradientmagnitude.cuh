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

#include <thrust/reduce.h>

namespace ExposureRender
{

KERNEL void KrnlComputeGradientMagnitudeVolume(unsigned short* pGradientMagnitude, int Width, int Height, int Depth)
{
	/*
	const int X = blockIdx.x * blockDim.x + threadIdx.x;
	const int Y	= blockIdx.y * blockDim.y + threadIdx.y;
	const int Z	= blockIdx.z * blockDim.z + threadIdx.z;
	
	if (X >= Width || Y >= Height || Z >= Depth)
		return;
	
	const Vec3f P = gpTracer->Volume.MinAABB + gpTracer->Volume.Size * (Vec3f((float)X + 0.5f, (float)Y + 0.5f, (float)Z + 0.5f) * gpTracer->Volume.InvResolution);

	int ID = X + Y * Width + Z * (Width * Height);

	pGradientMagnitude[ID] = GradientMagnitude(P);
	*/
}

void ComputeGradientMagnitudeVolume(int Extent[3], float& MaximumGradientMagnitude)
{
	const dim3 BlockDim(8, 8, 8);
	const dim3 GridDim((int)ceilf((float)Extent[0] / (float)BlockDim.x), (int)ceilf((float)Extent[1] / (float)BlockDim.y), (int)ceilf((float)Extent[2] / (float)BlockDim.z));

	unsigned short* pGradientMagnitude = NULL;

	// Allocate temporary linear memory for computation
	Cuda::Allocate(pGradientMagnitude, Extent[0] * Extent[1] * Extent[2]);

	// Execute gradient computation kernel
	LAUNCH_CUDA_KERNEL((KrnlComputeGradientMagnitudeVolume<<<GridDim, BlockDim>>>(pGradientMagnitude, Extent[0], Extent[1], Extent[2])));
	
	// Create thrust device pointer
	thrust::device_ptr<unsigned short> DevicePtr(pGradientMagnitude); 

	// Reduce the volume to a maximum gradient magnitude
	float Result = 0.0f;
	Result = thrust::reduce(DevicePtr, DevicePtr + Extent[0] * Extent[1] * Extent[2], Result, thrust::maximum<unsigned short>());
	
	// Free temporary memory
	Cuda::Free(pGradientMagnitude);

	// Set result
	MaximumGradientMagnitude = Result;
}

}