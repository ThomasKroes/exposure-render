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