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

#include "color.h"
#include "geometry.h"
#include "filter.h"

namespace ExposureRender
{

#define KRNL_BILATERAL_FILTER_BLOCK_W		16
#define KRNL_BILATERAL_FILTER_BLOCK_H		8
#define KRNL_BILATERAL_FILTER_BLOCK_SIZE	KRNL_BILATERAL_FILTER_BLOCK_W * KRNL_BILATERAL_FILTER_BLOCK_H

// http://code.google.com/p/bilateralfilter/source/browse/trunk/BilateralFilter.cpp?r=3
// http://code.google.com/p/bilateralfilter/source/browse/trunk/main.cpp

HOST_DEVICE inline float GetSpatialWeight(BilateralFilter& Filter, const int& IDx, const int& KernelX)
{
	return Filter.KernelD[Filter.KernelRadius + KernelX - IDx];
}

HOST_DEVICE inline float GaussianSimilarity(BilateralFilter& Filter, const ColorRGBf& A, const ColorRGBf& B)
{
	return Filter.GaussSimilarity[(int)fabs(LuminanceFromRGB(A[0], A[1], A[2]) - LuminanceFromRGB(B[0], B[1], B[2]))];//(int)floorf(A[0] - B[0])];
}

HOST_DEVICE ColorRGBf ToColorRGBf(ColorRGBAuc Color)
{
	return ColorRGBf(Color[0], Color[1], Color[2]);
}

HOST_DEVICE inline float FilterWeight(const int& IDx, const int& KernelX, const ColorRGBf& KernelPosColor, const ColorRGBf& CenterColor)
{
	return GetSpatialWeight(IDx, KernelX) * GaussianSimilarity(KernelPosColor, CenterColor);
}

KERNEL void KrnlBilateralFilterHorizontal(ColorRGBAuc* pIn, ColorRGBAuc* pOut, int Width, int Height)
{
	KERNEL_2D(gpTracer->FrameBuffer.Resolution[0], gpTracer->FrameBuffer.Resolution[1])

	// Compute kernel spatial range, taking into account the image boundaries
	__shared__ int Range[KRNL_BILATERAL_FILTER_BLOCK_SIZE][2];
	
	__syncthreads();

	BilateralFilter& Filter = gpTracer->RenderSettings.Filtering.PostProcessingFilter;

	Range[IDt][0] = max(0, IDx - Filter.KernelRadius);
	Range[IDt][1] = min(IDx + Filter.KernelRadius, Width - 1);

	__shared__ float Weight[KRNL_BILATERAL_FILTER_BLOCK_SIZE];
	__shared__ float TotalWeight[KRNL_BILATERAL_FILTER_BLOCK_SIZE];
	
	__syncthreads();

	Weight[IDt]			= 0.0f;
	TotalWeight[IDt]	= 0.0f;
	
	// Filter accumulated sum color
	__shared__ ColorRGBf Sum[KRNL_BILATERAL_FILTER_BLOCK_SIZE];

	__syncthreads();

	Sum[IDt].Black();

	// Kernel center pixel color
	ColorRGBf CenterColor = ToColorRGBf(pIn[IDk]);

	// Execute kernel
	for (int x = Range[IDt][0]; x < Range[IDt][1]; x++)
	{
		// Get color at kernel position
		ColorRGBf KernelPosColor = ToColorRGBf(pIn[IDy * Width + x]);

		// Compute composite weight
		Weight[IDt] = FilterWeight(IDx, x, KernelPosColor, CenterColor);

		// Compute total weight for normalization
		TotalWeight[IDt] += Weight[IDt];
		
		// Apply weight
		ColorRGBf C = KernelPosColor * Weight[IDt];

		// Compute sum color
		Sum[IDt] += C;
    }

	__syncthreads();

    if (TotalWeight[IDt] > 0.0f)
	{
		pOut[IDk][0] = Clamp(Sum[IDt][0] / TotalWeight[IDt], 0.0f, 255.0f);
		pOut[IDk][1] = Clamp(Sum[IDt][1] / TotalWeight[IDt], 0.0f, 255.0f);
		pOut[IDk][2] = Clamp(Sum[IDt][2] / TotalWeight[IDt], 0.0f, 255.0f);
		pOut[IDk][3] = 255;
	}
	else
	{
		pOut[IDk][0] = 0;
		pOut[IDk][1] = 0;
		pOut[IDk][2] = 0;
		pOut[IDk][3] = 255;
	}
}

KERNEL void KrnlBilateralFilterVertical(ColorRGBAuc* pIn, ColorRGBAuc* pOut, int Width, int Height)
{
	KERNEL_2D(gpTracer->FrameBuffer.Resolution[0], gpTracer->FrameBuffer.Resolution[1])

	// Compute kernel spatial range, taking into account the image boundaries
	__shared__ int Range[KRNL_BILATERAL_FILTER_BLOCK_SIZE][2];
	
	__syncthreads();

	BilateralFilter& Filter = gpTracer->RenderSettings.Filtering.PostProcessingFilter;

	Range[IDt][0] = max(0, IDy - Filter.KernelRadius);
	Range[IDt][1] = min(IDy + Filter.KernelRadius, Height - 1);

	__shared__ float Weight[KRNL_BILATERAL_FILTER_BLOCK_SIZE];
	__shared__ float TotalWeight[KRNL_BILATERAL_FILTER_BLOCK_SIZE];
	
	__syncthreads();

	Weight[IDt]			= 0.0f;
	TotalWeight[IDt]	= 0.0f;
	
	// Filter accumulated sum color
	__shared__ ColorRGBf Sum[KRNL_BILATERAL_FILTER_BLOCK_SIZE];

	__syncthreads();

	Sum[IDt].Black();

	// Kernel center pixel color
	ColorRGBf CenterColor = ToColorRGBf(pIn[IDk]);

	// Execute kernel
	for (int y = Range[IDt][0]; y < Range[IDt][1]; y++)
	{
		// Get color at kernel position
		ColorRGBf KernelPosColor = ToColorRGBf(pIn[y * Width + IDx]);

		// Compute composite weight
		Weight[IDt] = FilterWeight(IDy, y, KernelPosColor, CenterColor);

		// Compute total weight for normalization
		TotalWeight[IDt] += Weight[IDt];
		
		// Apply weight
		ColorRGBf C = KernelPosColor * Weight[IDt];

		// Compute sum color
		Sum[IDt] += C;
    }

	__syncthreads();

	if (TotalWeight[IDt] > 0.0f)
	{
		pOut[IDk][0] = Clamp(Sum[IDt][0] / TotalWeight[IDt], 0.0f, 255.0f);
		pOut[IDk][1] = Clamp(Sum[IDt][1] / TotalWeight[IDt], 0.0f, 255.0f);
		pOut[IDk][2] = Clamp(Sum[IDt][2] / TotalWeight[IDt], 0.0f, 255.0f);
		pOut[IDk][3] = 255;
	}
	else
	{
		pOut[IDk][0] = 0;
		pOut[IDk][1] = 0;
		pOut[IDk][2] = 0;
		pOut[IDk][3] = 255;
	}
}

void FilterBilateral(ColorRGBAuc* pImage, ColorRGBAuc* pTemp, ColorRGBAuc* pOut, int Width, int Height)
{
	const dim3 BlockDim(KRNL_BILATERAL_FILTER_BLOCK_W, KRNL_BILATERAL_FILTER_BLOCK_H);
	const dim3 GridDim((int)ceilf((float)Width / (float)BlockDim.x), (int)ceilf((float)Height / (float)BlockDim.y));

	LAUNCH_CUDA_KERNEL_TIMED((KrnlBilateralFilterHorizontal<<<GridDim, BlockDim>>>(pImage, pTemp, Width, Height)), "Bilateral filter (Horizontal)");
	LAUNCH_CUDA_KERNEL_TIMED((KrnlBilateralFilterVertical<<<GridDim, BlockDim>>>(pTemp, pOut, Width, Height)), "Bilateral filter (Vertical)");
}

}