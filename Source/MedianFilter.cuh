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

#include "Color.cuh"
#include "Geometry.cuh"
#include "Filter.cuh"

// Code from: http://www.powershow.com/view/26021-YWQxN/CUDA_ITK_flash_ppt_presentation

namespace ExposureRender
{

#define KRNL_MEDIAN_FILTER_BLOCK_W		8
#define KRNL_MEDIAN_FILTER_BLOCK_H		8
#define KRNL_MEDIAN_FILTER_BLOCK_SIZE	KRNL_MEDIAN_FILTER_BLOCK_W * KRNL_MEDIAN_FILTER_BLOCK_H

#define MEDIAN_FILTER_KERNEL_RADIUS		2
#define MEDIAN_FILTER_KERNEL_WIDTH		2 * MEDIAN_FILTER_KERNEL_RADIUS + 1
#define MEDIAN_FILTER_KERNEL_SIZE		MEDIAN_FILTER_KERNEL_WIDTH * MEDIAN_FILTER_KERNEL_WIDTH

KERNEL void KrnlMedianFilter(ColorRGBAuc* pInput, ColorRGBAuc* pOut, unsigned int Width, unsigned int Height)
{
	// Indexing
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y 	= blockIdx.y * blockDim.y + threadIdx.y;
	const int PID	= Y * Width + X;

	// Exit if beyond image boundaries
	if (X >= Width || Y >= Height)
		return;

	int Range[2][2];
	
	Range[0][0] = max((int)ceilf(X - MEDIAN_FILTER_KERNEL_RADIUS), 0);
	Range[0][1] = min((int)floorf(X + MEDIAN_FILTER_KERNEL_RADIUS), Width - 1);
	Range[1][0] = max((int)ceilf(Y - MEDIAN_FILTER_KERNEL_RADIUS), 0);
	Range[1][1] = min((int)floorf(Y + MEDIAN_FILTER_KERNEL_RADIUS), Height - 1);

	ColorRGBAuc Kernel[MEDIAN_FILTER_KERNEL_SIZE];

	int KernelSize = 0;

	for (int x = Range[0][0]; x < Range[0][1]; x++)
	{
		for (int y = Range[1][0]; y < Range[1][1]; y++)
		{
			Kernel[KernelSize] = pInput[y * Width + x];
			KernelSize++;
		}
	}

	for (int c = 0; c < 3; c++)
	{
		unsigned char Min = 0;
		unsigned char Max = 255;

		float Pivot = 0.5f * (Min + Max);

		for (int i = 0; i < 8; i++)
		{
			int Count = 0;

			for (int j = 0; j < KernelSize; j++)
			{
				if (Kernel[j][c] > Pivot)
					Count++;
			}

			if (Count < KernelSize / 2)
				Max = floorf(Pivot);
			else
				Min = ceilf(Pivot);

			Pivot = 0.5f * (Min + Max);
		}

		pOut[PID][c] = Pivot;
	}
}

void MedianFilter(ColorRGBAuc* pImage, ColorRGBAuc* pOut, int Width, int Height)
{
	const dim3 BlockDim(KRNL_MEDIAN_FILTER_BLOCK_W, KRNL_MEDIAN_FILTER_BLOCK_H);
	const dim3 GridDim((int)ceilf((float)Width / (float)BlockDim.x), (int)ceilf((float)Height / (float)BlockDim.y));

	LAUNCH_CUDA_KERNEL_TIMED((KrnlMedianFilter<<<GridDim, BlockDim>>>(pImage, pOut, Width, Height)), "Median filter");
}

}