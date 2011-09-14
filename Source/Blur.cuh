#pragma once

#include "Geometry.h"
#include "Filter.h"

KERNEL void KrnlBlurXyzH(CColorXyz* pImage, CColorXyz* pTempImage, CResolution2D Resolution, CGaussianFilter GaussianFilter)
{
	const int X 	= (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global y
	const int Y		= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global x
	const int PID	= (Y * Resolution.GetResX()) + X;				// Get pixel ID	

	// Exit if beyond image boundaries
	if (X >= Resolution.GetResX() || Y >= Resolution.GetResY())
		return;

	// Compute filter extent
	const int X0 = max((int)ceilf(X - GaussianFilter.xWidth), 0);
	const int X1 = min((int)floorf(X + GaussianFilter.xWidth), (int)Resolution.GetResX());

	// Accumulated color
	CColorXyz Sum;

	// Weights
	float FW = 1.0f, SumW = 0.0f;

	for (int x = X0; x <= X1; x++)
	{
		// Compute filter weight
		FW = GaussianFilter.Evaluate(fabs((float)(x - X) / (0.5f * GaussianFilter.xWidth)), 0.0f);

		Sum		+= FW * pImage[(Y * (int)Resolution.GetResX()) + x];
		SumW	+= FW;
	}

	__syncthreads();

	// Write to temporary image
	pTempImage[PID] = Sum / SumW;
}

// ToDo: Add description
KERNEL void KrnlBlurXyzV(CColorXyz* pImage, CColorXyz* pTempImage, CResolution2D Resolution, CGaussianFilter GaussianFilter)
{
	const int X 	= (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global y
	const int Y		= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global x
	const int PID	= (Y * Resolution.GetResX()) + X;				// Get pixel ID	

	// Exit if beyond image boundaries
	if (X >= Resolution.GetResX() || Y >= Resolution.GetResY())
		return;

	// Compute filter extent
	const int Y0 = max((int)ceilf (Y - GaussianFilter.yWidth), 0);
	const int Y1 = min((int)floorf(Y + GaussianFilter.yWidth), (int)Resolution.GetResY() - 1);

	// Accumulated color
	CColorXyz Sum;

	// Weights
	float FW = 1.0f, SumW = 0.0f;

	for (int y = Y0; y <= Y1; y++)
	{
		// Compute filter weight
		FW = GaussianFilter.Evaluate(0.0f, fabs((float)(y - Y) / (0.5f * GaussianFilter.yWidth)));

		Sum		+= FW * pTempImage[(y * (int)Resolution.GetResX()) + X];
		SumW	+= FW;
	}

	__syncthreads();

	// Write to image
	pImage[PID]	= Sum / SumW;
}

// ToDo: Add description
void BlurImageXyz(CColorXyz* pImage, CColorXyz* pTempImage, const CResolution2D& Resolution, const float& Radius)
{
	const dim3 KernelBlock(32, 8);
	const dim3 KernelGrid((int)ceilf((float)Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)Resolution.GetResY() / (float)KernelBlock.y));

	// Create gaussian filter
	CGaussianFilter GaussianFilter(2.5f * Radius, 2.5f * Radius, 1.0f);

	KrnlBlurXyzH<<<KernelGrid, KernelBlock>>>(pImage, pTempImage, Resolution, GaussianFilter); 
	KrnlBlurXyzV<<<KernelGrid, KernelBlock>>>(pImage, pTempImage, Resolution, GaussianFilter); 
}