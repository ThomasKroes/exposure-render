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

#include "Buffer.cuh"

#define KRNL_BLUR_BLOCK_W		16
#define KRNL_BLUR_BLOCK_H		8
#define KRNL_BLUR_BLOCK_SIZE	KRNL_BLUR_BLOCK_W * KRNL_BLUR_BLOCK_H

KERNEL void KrnlBlurEstimateH(FrameBuffer* pFrameBuffer)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

	if (X >= pFrameBuffer->Resolution[0] || Y >= pFrameBuffer->Resolution[1])
		return;

	const int X0 = max((int)ceilf(X - gBlur.FilterWidth), 0);
	const int X1 = min((int)floorf(X + gBlur.FilterWidth), (int)gCamera.FilmWidth - 1);

	ColorXYZAf Sum;

	__shared__ float FW[KRNL_BLUR_BLOCK_SIZE];
	__shared__ float SumW[KRNL_BLUR_BLOCK_SIZE];

	__syncthreads();

	FW[TID]		= 0.0f;
	SumW[TID]	= 0.0f;

	for (int x = X0; x <= X1; x++)
	{
		FW[TID] = gBlur.FilterWeights[(int)fabs((float)x - X)];

		Sum			+= pFrameBuffer->CudaFrameEstimateXyza.Get(x, Y) * FW[TID];
		SumW[TID]	+= FW[TID];
	}

	if (SumW[TID] > 0.0f)
		pFrameBuffer->CudaFrameBlurXyza.Set(Sum / SumW[TID], X, Y);
	else
		pFrameBuffer->CudaFrameBlurXyza.Set(ColorXYZAf(0.0f), X, Y);
}

KERNEL void KrnlBlurEstimateV(FrameBuffer* pFrameBuffer)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

	if (X >= pFrameBuffer->Resolution[0] || Y >= pFrameBuffer->Resolution[1])
		return;

	const int Y0 = max((int)ceilf (Y - gBlur.FilterWidth), 0);
	const int Y1 = min((int)floorf(Y + gBlur.FilterWidth), gCamera.FilmHeight - 1);

	ColorXYZAf Sum;

	__shared__ float FW[KRNL_BLUR_BLOCK_SIZE];
	__shared__ float SumW[KRNL_BLUR_BLOCK_SIZE];

	__syncthreads();

	FW[TID]		= 0.0f;
	SumW[TID]	= 0.0f;

	for (int y = Y0; y <= Y1; y++)
	{
		FW[TID] = gBlur.FilterWeights[(int)fabs((float)y - Y)];

		Sum			+= pFrameBuffer->CudaFrameBlurXyza.Get(X, y) * FW[TID];
		SumW[TID]	+= FW[TID];
	}

	if (SumW[TID] > 0.0f)
		pFrameBuffer->CudaFrameEstimateXyza.Set(Sum / SumW[TID], X, Y);
	else
		pFrameBuffer->CudaFrameEstimateXyza.Set(ColorXYZAf(0.0f), X, Y);
}

void BlurEstimate(FrameBuffer* pFrameBuffer, int Width, int Height)
{
	const dim3 BlockDim(KRNL_BLUR_BLOCK_W, KRNL_BLUR_BLOCK_H);
	const dim3 GridDim((int)ceilf((float)Width / (float)BlockDim.x), (int)ceilf((float)Height / (float)BlockDim.y));

	KrnlBlurEstimateH<<<GridDim, BlockDim>>>(pFrameBuffer);
	cudaThreadSynchronize();
	
	KrnlBlurEstimateV<<<GridDim, BlockDim>>>(pFrameBuffer);
	cudaThreadSynchronize();
}