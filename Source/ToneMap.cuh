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

#include "Geometry.cuh"

#define KRNL_TONE_MAP_BLOCK_W		16 
#define KRNL_TONE_MAP_BLOCK_H		8
#define KRNL_TONE_MAP_BLOCK_SIZE	KRNL_TONE_MAP_BLOCK_W * KRNL_TONE_MAP_BLOCK_H

// http://code.google.com/p/bilateralfilter/source/browse/trunk/BilateralFilter.cpp?r=3
// http://code.google.com/p/bilateralfilter/source/browse/trunk/main.cpp

KERNEL void KrnlToneMap(FrameBuffer* pFrameBuffer)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= pFrameBuffer->Resolution[0] || Y >= pFrameBuffer->Resolution[1])
		return;

	const ColorRGBuc L1 = ToneMap(pFrameBuffer->CudaRunningEstimateXyza.Get(X, Y));

	pFrameBuffer->CudaDisplayEstimateA(X, Y)[0] = L1[0];
	pFrameBuffer->CudaDisplayEstimateA(X, Y)[1] = L1[1];
	pFrameBuffer->CudaDisplayEstimateA(X, Y)[2] = L1[2];
	pFrameBuffer->CudaDisplayEstimateA(X, Y)[3] = pFrameBuffer->CudaRunningEstimateXyza(X, Y)[3] * 255.0f;
}

KERNEL void KrnlFilter(FrameBuffer* pFrameBuffer)
{
	const int X = blockIdx.x * blockDim.x + threadIdx.x;
	const int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= pFrameBuffer->Resolution[0] || Y >= pFrameBuffer->Resolution[1])
		return;

	int Width = 5;

	int Range[2][2] =
	{
		{ max(0, X - Width), min(X + Width, pFrameBuffer->Resolution[0] - 1) },
		{ max(0, Y - Width), min(Y + Width, pFrameBuffer->Resolution[1] - 1) },
	};

	float SumWeight = 0.0f;

	float Sum[3] = { 0.0f, 0.0f, 0.0f };

	for (int x = Range[0][0]; x < Range[0][1]; x++)
	{
		for (int y = Range[1][0]; y < Range[1][1]; y++)
		{
			float RGB[3] =
			{
				pFrameBuffer->CudaDisplayEstimateA(x, y)[0],
				pFrameBuffer->CudaDisplayEstimateA(x, y)[1],
				pFrameBuffer->CudaDisplayEstimateA(x, y)[2]
			};

			float NormLuminance = RGB[0] + RGB[1] + RGB[2];
			NormLuminance /= 3.0f;
			NormLuminance /= 255.0f;

			NormLuminance = 1.0f;

			Sum[0] += NormLuminance * RGB[0];
			Sum[1] += NormLuminance * RGB[1];
			Sum[2] += NormLuminance * RGB[2];

			SumWeight += 1;
		}
	}

	pFrameBuffer->CudaDisplayEstimateB(X, Y)[0] = Sum[0] / SumWeight;
	pFrameBuffer->CudaDisplayEstimateB(X, Y)[1] = Sum[1] / SumWeight;
	pFrameBuffer->CudaDisplayEstimateB(X, Y)[2] = Sum[2] / SumWeight;
	pFrameBuffer->CudaDisplayEstimateB(X, Y)[3] = pFrameBuffer->CudaDisplayEstimateA(X, Y)[3];
}

CD int gKernelRadius;
CD float gKernelD[9][9];
CD float gGaussSimilarity[256];

HOST_DEVICE inline float GetSpatialWeight(int m, int n,int i,int j)
{
	return gKernelD[(int)(i-m + gKernelRadius)][(int)(j-n + gKernelRadius)];
}

HOST_DEVICE inline float Gauss(float sigma, int x, int y)
{
	return expf(-((x * x + y * y) / (2 * sigma * sigma)));
}

HOST_DEVICE inline float GaussianSimilarity(int p, int s)
{
	return gGaussSimilarity[abs(p-s)];
}

KERNEL void KrnlBilateralFilter(FrameBuffer* pFrameBuffer)
{
	const int X = blockIdx.x * blockDim.x + threadIdx.x;
	const int Y = blockIdx.y * blockDim.y + threadIdx.y;

	int Range[2][2] =
	{
		{ max(0, X - gKernelRadius), min(X + gKernelRadius, pFrameBuffer->Resolution[0] - 1) },
		{ max(0, Y - gKernelRadius), min(Y + gKernelRadius, pFrameBuffer->Resolution[1] - 1) },
	};

    float weight = 0.0f, totalWeight = 0.0f;

	ColorXYZAf sum;

	ColorXYZAf intensityCenter;

	intensityCenter[0] = pFrameBuffer->CudaDisplayEstimateA(X, Y)[0];
	intensityCenter[1] = pFrameBuffer->CudaDisplayEstimateA(X, Y)[1];
	intensityCenter[2] = pFrameBuffer->CudaDisplayEstimateA(X, Y)[2];
	intensityCenter[3] = pFrameBuffer->CudaDisplayEstimateA(X, Y)[3];

	for (int m = Range[0][0]; m < Range[0][1]; m++)
	{
		for (int n = Range[1][0]; n < Range[1][1]; n++)
		{
			ColorXYZAf intensityKernelPos;

			intensityKernelPos[0] = pFrameBuffer->CudaDisplayEstimateA(m, n)[0];
			intensityKernelPos[1] = pFrameBuffer->CudaDisplayEstimateA(m, n)[1];
			intensityKernelPos[2] = pFrameBuffer->CudaDisplayEstimateA(m, n)[2];
			intensityKernelPos[3] = pFrameBuffer->CudaDisplayEstimateA(m, n)[3];

			weight = GetSpatialWeight(m,n,X,Y) * GaussianSimilarity(intensityKernelPos[0], intensityCenter[0]);
			totalWeight += weight;
			sum += (weight * intensityKernelPos);
		}
    }
    pFrameBuffer->CudaDisplayEstimateB(X, Y)[0] = sum[0] / totalWeight;
	pFrameBuffer->CudaDisplayEstimateB(X, Y)[1] = sum[1] / totalWeight;
	pFrameBuffer->CudaDisplayEstimateB(X, Y)[2] = sum[2] / totalWeight;
	pFrameBuffer->CudaDisplayEstimateB(X, Y)[3] = sum[3] / totalWeight;
}

void BilaterialFilter(FrameBuffer* pFrameBuffer, int Width, int Height)
{
	float sigmaD = 10.0f, float sigmaR = 6.0f;
	float twoSigmaRSquared = 2 * sigmaR * sigmaR;
	int kernelRadius = 4;
	int kernelSize = kernelRadius * 2 + 1;

	int center = (kernelSize - 1) / 2;
	
	float kernelD[9][9];

	for (int x = -center; x < -center + kernelSize; x++)
	{
		for (int y = -center; y < -center + kernelSize; y++)
		{
			kernelD[x + center][y + center] = Gauss(sigmaD, x, y);
		}
	}

	float gaussSimilarity[256];
	
	for (int i = 0; i < 256; i++)
	{
		gaussSimilarity[i] = expf((double)-((i) / twoSigmaRSquared));
	}

	cudaMemcpyToSymbol("gKernelRadius", &kernelRadius, sizeof(int));
	cudaMemcpyToSymbol("gKernelD", kernelD, kernelSize * sizeof(float));
	cudaMemcpyToSymbol("gaussSimilarity", gaussSimilarity, 256 * sizeof(float));

	const dim3 BlockDim(KRNL_TONE_MAP_BLOCK_W, KRNL_TONE_MAP_BLOCK_H);
	const dim3 GridDim((int)ceilf((float)Width / (float)BlockDim.x), (int)ceilf((float)Height / (float)BlockDim.y));

	KrnlBilateralFilter<<<GridDim, BlockDim>>>(pFrameBuffer);
}

KERNEL void KrnlBlend(FrameBuffer* pFrameBuffer)
{
	const int X = blockIdx.x * blockDim.x + threadIdx.x;
	const int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= pFrameBuffer->Resolution[0] || Y >= pFrameBuffer->Resolution[1])
		return;

	pFrameBuffer->CudaDisplayEstimateA(X, Y) = Lerp(pFrameBuffer->CudaDisplayEstimateA(X, Y), pFrameBuffer->CudaDisplayEstimateB(X, Y), expf(-(float)gScattering.NoIterations / 10.0f));
}

void PostProcess(FrameBuffer* pFrameBuffer, int Width, int Height)
{
	const dim3 BlockDim(KRNL_TONE_MAP_BLOCK_W, KRNL_TONE_MAP_BLOCK_H);
	const dim3 GridDim((int)ceilf((float)Width / (float)BlockDim.x), (int)ceilf((float)Height / (float)BlockDim.y));

	KrnlToneMap<<<GridDim, BlockDim>>>(pFrameBuffer);
	BilaterialFilter(pFrameBuffer, Width, Height);
//	KrnlFilter<<<GridDim, BlockDim>>>(pFrameBuffer);
	KrnlBlend<<<GridDim, BlockDim>>>(pFrameBuffer);
}