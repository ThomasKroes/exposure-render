#pragma once

#include "Geometry.h"
#include "Variance.h"

__constant__ float SumVariance = 0;

KERNEL void KrnlComputeVariance(int Width, int Height, CColorXyz* gpEstXyz, CColorXyz* pAccEstXyz, float N, float Exposure, CColorRgbaLdr* pPixels)
{
	const int X 	= (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global Y
	const int Y		= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global X
	const int PID	= (Y * Width) + X;								// Get pixel ID	

	// Exit if beyond image boundaries
	if (X >= Width || Y >= Height)
		return;

	const int MinX = gridDim.x * blockDim.x;
	const int MinY = gridDim.y * blockDim.y;

	__shared__ float LocalSum[8];

	if (threadIdx.x == 0)
	{
		for (int x = MinX; x < MinX + blockDim.x; x++)
		{
// 			LocalSum[threadIdx.y] += ;
		}
	}

//	atomicAdd(&SumVariance, 1.0f);


	__syncthreads();
}

void ComputeVariance(int Width, int Height, CColorXyz* pEstFrameXyz, CColorXyz* pAccEstXyz, float N, float Exposure, CColorRgbaLdr* pPixels)
{
	const dim3 KernelBlock(16, 8);
	const dim3 KernelGrid((int)ceilf((float)Width / (float)KernelBlock.x), (int)ceilf((float)Height / (float)KernelBlock.y));

	KrnlComputeVariance<<<KernelGrid, KernelBlock>>>(Width, Height, pEstFrameXyz, pAccEstXyz, N, Exposure, pPixels); 
}