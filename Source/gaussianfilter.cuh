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

#include "color.h"
#include "filter.h"
#include "tracer.h"

namespace ExposureRender
{

/*
KERNEL void KrnlGaussianFilterHorizontal(ColorXYZAf* pIn, ColorXYZAf* pOut)
{
	KERNEL_2D(gpTracer->FrameBuffer.Resolution[0], gpTracer->FrameBuffer.Resolution[1])

	// Compute kernel spatial range, taking into account the image boundaries
	__shared__ int Range[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE][2];
	
	__syncthreads();

	const GaussianFilter& Filter = gpTracer->FrameEstimateFilter;

	Range[IDt][0] = max((int)ceilf(IDx - Filter.KernelRadius), 0);
	Range[IDt][1] = min((int)floorf(IDx + Filter.KernelRadius), gpTracer->FrameBuffer.Resolution[0] - 1);

	// Filter accumulated sum color
	__shared__ ColorXYZAf Sum[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];

	__syncthreads();

	Sum[IDt] = ColorXYZAf(0.0f, 0.0f, 0.0f, 0.0f);

	__shared__ float Weight[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];
	__shared__ float TotalWeight[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];

	__syncthreads();

	Weight[IDt]			= 0.0f;
	TotalWeight[IDt]	= 0.0f;

	// Execute kernel
	for (int x = Range[IDt][0]; x <= Range[IDt][1]; x++)
	{
		Weight[IDt]			= Filter.KernelD[Filter.KernelRadius - (x - IDx)];
		Sum[IDt]			+= pIn[IDy * gpTracer->FrameBuffer.Resolution[0] + x] * Weight[IDt];
		TotalWeight[IDt]	+= Weight[IDt];
	}

	__syncthreads();

	if (TotalWeight[IDt] > 0.0f)
		pOut[IDk] = Sum[IDt] / TotalWeight[IDt];
	else
		pOut[IDk] = ColorXYZAf(0.0f);
}

KERNEL void KrnlGaussianFilterVertical(ColorXYZAf* pIn, ColorXYZAf* pOut)
{
	KERNEL_2D(gpTracer->FrameBuffer.Resolution[0], gpTracer->FrameBuffer.Resolution[1])

	// Compute kernel spatial range, taking into account the image boundaries
	__shared__ int Range[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE][2];
	
	__syncthreads();

	const GaussianFilter& Filter = gpTracer->FrameEstimateFilter;

	Range[IDt][0] = max((int)ceilf(IDy - Filter.KernelRadius), 0);
	Range[IDt][1] = min((int)floorf(IDy + Filter.KernelRadius), gpTracer->FrameBuffer.Resolution[1] - 1);

	// Filter accumulated sum color
	__shared__ ColorXYZAf Sum[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];

	__syncthreads();

	Sum[IDt] = ColorXYZAf(0.0f, 0.0f, 0.0f, 0.0f);

	__shared__ float Weight[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];
	__shared__ float TotalWeight[KRNL_GAUSSIAN_FILTER_BLOCK_SIZE];

	__syncthreads();

	Weight[IDt]			= 0.0f;
	TotalWeight[IDt]	= 0.0f;

	// Execute kernel
	for (int y = Range[IDt][0]; y <= Range[IDt][1]; y++)
	{
		Weight[IDt]			= Filter.KernelD[Filter.KernelRadius - (y - IDy)];
		Sum[IDt]			+= pIn[y * gpTracer->FrameBuffer.Resolution[0] + IDx] * Weight[IDt];
		TotalWeight[IDt]	+= Weight[IDt];
	}

	__syncthreads();

	if (TotalWeight[IDt] > 0.0f)
		pOut[IDk] = Sum[IDt] / TotalWeight[IDt];
	else
		pOut[IDk] = ColorXYZAf(0.0f);
}

void FilterGaussian(ColorXYZAf* pImage, ColorXYZAf* pTemp, int Width, int Height)
{
	LAUNCH_DIMENSIONS(Width, Height, 1, 16, 8, 1)
	LAUNCH_CUDA_KERNEL_TIMED((KrnlGaussianFilterHorizontal<<<GridDim, BlockDim>>>(pImage, pTemp)), "Gaussian filter (Horizontal)");
	LAUNCH_CUDA_KERNEL_TIMED((KrnlGaussianFilterVertical<<<GridDim, BlockDim>>>(pTemp, pImage)), "Gaussian filter (Vertical)");
}
*/

CD float gFilterWeights[5][5] =
{
	{ 1.0f, 4.0f, 7.0f, 4.0f, 1.0f },
	{ 4.0f, 16.0f, 26.0f, 16.0f, 4.0f },
	{ 7.0f, 26.0f, 41.0f, 26, 7.0f },
	{ 4.0f, 16.0f, 26.0f, 16.0f, 4.0f },
	{ 1.0f, 4.0f, 7.0f, 4.0f, 1.0f }
};

KERNEL void KrnlGaussianFilter(ColorXYZAf* pIn, ColorXYZAf* pOut)
{
	KERNEL_2D(gpTracer->FrameBuffer.Resolution[0], gpTracer->FrameBuffer.Resolution[1])

	int Range[2][2];

	Range[0][0] = max((int)ceilf(IDx - 2), 0);
	Range[0][1] = min((int)floorf(IDx + 2), gpTracer->FrameBuffer.Resolution[0] - 1);
	Range[1][0] = max((int)ceilf(IDy - 2), 0);
	Range[1][1] = min((int)floorf(IDy + 2), gpTracer->FrameBuffer.Resolution[1] - 1);

	ColorXYZAf Sum		= ColorXYZAf::Black();
	float Weight		= 0.0f;
	float TotalWeight	= 0.0f;

	for (int y = Range[1][0]; y <= Range[1][1]; y++)
	{
		for (int x = Range[0][0]; x <= Range[0][1]; x++)
		{
			Weight		= 1.0f;//gFilterWeights[2 - (y - IDy)][2 - (x - IDx)];
			Sum			+= pIn[y * gpTracer->FrameBuffer.Resolution[0] + x] * Weight;
			TotalWeight	+= Weight;
		}
	}

	if (TotalWeight > 0.0f)
		pOut[IDk] = Sum / TotalWeight;
	else
		pOut[IDk] = ColorXYZAf::Black();
}

void FilterGaussian(Tracer& Tracer)
{
	LAUNCH_DIMENSIONS(Tracer.FrameBuffer.Resolution[0], Tracer.FrameBuffer.Resolution[1], 1, 8, 8, 1)
	LAUNCH_CUDA_KERNEL_TIMED((KrnlGaussianFilter<<<GridDim, BlockDim>>>(&Tracer.FrameBuffer.FrameEstimate[0], &Tracer.FrameBuffer.FrameEstimateTemp[0])), "Gaussian filter (Horizontal)");

	Tracer.FrameBuffer.FrameEstimate = Tracer.FrameBuffer.FrameEstimateTemp;
}

}
