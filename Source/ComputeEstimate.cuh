#pragma once

#include "Geometry.h"

KERNEL void KrnlComputeEstimate(int Width, int Height, CColorXyz* gpEstFrameXyz, CColorXyz* pAccEstXyz, float N, float Exposure, unsigned char* pPixels)
{
	const int X 	= (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global Y
	const int Y		= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global X
	const int PID	= (Y * Width) + X;								// Get pixel ID	

	// Exit if beyond image boundaries
	if (X >= Width || Y >= Height)
		return;

	pAccEstXyz[PID] += gpEstFrameXyz[PID];

	const CColorXyz L = pAccEstXyz[PID] / (float)__max(1.0f, N);

	float InvGamma = 1.0f / 2.2f;

	CColorRgbHdr RgbHdr;
	
	RgbHdr.FromXYZ(L.c[0], L.c[1], L.c[2]);

	RgbHdr.r = Clamp(1.0f - expf(-(RgbHdr.r / Exposure)), 0.0, 1.0f);
	RgbHdr.g = Clamp(1.0f - expf(-(RgbHdr.g / Exposure)), 0.0, 1.0f);
	RgbHdr.b = Clamp(1.0f - expf(-(RgbHdr.b / Exposure)), 0.0, 1.0f);

	pPixels[(3 * (Y * Width + X)) + 0] = (unsigned char)Clamp((255.0f * powf(RgbHdr.r, InvGamma)), 0.0f, 255.0f);
	pPixels[(3 * (Y * Width + X)) + 1] = (unsigned char)Clamp((255.0f * powf(RgbHdr.g, InvGamma)), 0.0f, 255.0f);
	pPixels[(3 * (Y * Width + X)) + 2] = (unsigned char)Clamp((255.0f * powf(RgbHdr.b, InvGamma)), 0.0f, 255.0f);
}

void ComputeEstimate(int Width, int Height, CColorXyz* pEstFrameXyz, CColorXyz* pAccEstXyz, float N, float Exposure, unsigned char* pPixels)
{
	const dim3 KernelBlock(16, 8);
	const dim3 KernelGrid((int)ceilf((float)Width / (float)KernelBlock.x), (int)ceilf((float)Height / (float)KernelBlock.y));

	KrnlComputeEstimate<<<KernelGrid, KernelBlock>>>(Width, Height, pEstFrameXyz, pAccEstXyz, N, Exposure, pPixels); 
}