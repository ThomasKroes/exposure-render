#pragma once

#include "Geometry.h"
#include "Variance.h"

__constant__ float SumVariance = 0;

KERNEL void KrnlComputeVariance(int Width, int Height, CColorXyz* gpEstXyz, CColorXyz* pAccEstXyz, float N, float Exposure, CColorRgbaLdr* pPixels)
{
}

void ComputeVariance(int Width, int Height, CColorXyz* pEstFrameXyz, CColorXyz* pAccEstXyz, float N, float Exposure, CColorRgbaLdr* pPixels)
{
	const dim3 KernelBlock(16, 8);
	const dim3 KernelGrid((int)ceilf((float)Width / (float)KernelBlock.x), (int)ceilf((float)Height / (float)KernelBlock.y));

	KrnlComputeVariance<<<KernelGrid, KernelBlock>>>(Width, Height, pEstFrameXyz, pAccEstXyz, N, Exposure, pPixels); 
}