#pragma once

#include "Geometry.h"
#include "Filter.h"
#include "Scene.h"
#include "CudaUtilities.h"

#define KRNL_BLUR_BLOCK_W		16
#define KRNL_BLUR_BLOCK_H		8
#define KRNL_BLUR_BLOCK_SIZE	KRNL_BLUR_BLOCK_W * KRNL_BLUR_BLOCK_H

KERNEL void KrnlSharedBlurXyzH(void)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	const int X0 = max((int)ceilf(X - gFilterWidth), 0);
	const int X1 = min((int)floorf(X + gFilterWidth), (int)gFilmWidth - 1);

	CColorXyz Sum;

	__shared__ float FW[KRNL_BLUR_BLOCK_SIZE];
	__shared__ float SumW[KRNL_BLUR_BLOCK_SIZE];

	__syncthreads();

	FW[TID]		= 0.0f;
	SumW[TID]	= 0.0f;

	for (int x = X0; x <= X1; x++)
	{
		FW[TID] = 1.0f;//gFilterWeights[(int)fabs((float)x - X)];

		const float4 ColorXyza = tex2D(gTexFrameEstimateXyza, x, Y);

		Sum			+= FW[TID] * CColorXyz(ColorXyza.x, ColorXyza.y, ColorXyza.z);
		SumW[TID]	+= FW[TID];
	}

	Sum /= SumW[TID];

	surf2Dwrite(make_float4(Sum.c[0], Sum.c[1], Sum.c[2], 0.0f), gSurfFrameBlurXyza, X * sizeof(float4), Y);
}

KERNEL void KrnlSharedBlurXyzV(void)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	const int Y0 = max((int)ceilf (Y - gFilterWidth), 0);
	const int Y1 = min((int)floorf(Y + gFilterWidth), gFilmHeight - 1);

	CColorXyz Sum;

	__shared__ float FW[KRNL_BLUR_BLOCK_SIZE];
	__shared__ float SumW[KRNL_BLUR_BLOCK_SIZE];

	__syncthreads();

	FW[TID]		= 0.0f;
	SumW[TID]	= 0.0f;

	for (int y = Y0; y <= Y1; y++)
	{
		FW[TID] = 1.0f;//gFilterWeights[(int)fabs((float)y - Y)];

		const float4 ColorXyza = tex2D(gTexFrameBlurXyza, X, y);

		Sum			+= FW[TID] * CColorXyz(ColorXyza.x, ColorXyza.y, ColorXyza.z);
		SumW[TID]	+= FW[TID];
	}

	Sum /= SumW[TID];

	const float4 ColorXYZA = make_float4(Sum.c[0], Sum.c[1], Sum.c[2], 0.0f);
	surf2Dwrite(ColorXYZA, gSurfFrameEstimateXyza, X * sizeof(float4), Y);
}

KERNEL void KrnlBlurXyzH(void)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	const int X0 = max((int)ceilf(X - gFilterWidth), 0);
	const int X1 = min((int)floorf(X + gFilterWidth), (int)gFilmWidth - 1);

	CColorXyz Sum;

	float FW	= 0.0f;
	float SumW	= 0.0f;

	for (int x = X0; x <= X1; x++)
	{
		FW = 1.0f;//gFilterWeights[(int)fabs((float)x - X)];

		const float4 ColorXyza = tex2D(gTexFrameEstimateXyza, x + 0.5f, Y + 0.5f);

		Sum		+= FW * CColorXyz(ColorXyza.x, ColorXyza.y, ColorXyza.z);
		SumW	+= FW;
	}

	Sum /= SumW;

	surf2Dwrite(make_float4(Sum.c[0], Sum.c[1], Sum.c[2], 0.0f), gSurfFrameBlurXyza, X * sizeof(float4), Y);
}

KERNEL void KrnlBlurXyzV(void)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	const int Y0 = max((int)ceilf (Y - gFilterWidth), 0);
	const int Y1 = min((int)floorf(Y + gFilterWidth), gFilmHeight - 1);

	CColorXyz Sum;

	float FW	= 0.0f;
	float SumW	= 0.0f;

	for (int y = Y0; y <= Y1; y++)
	{
		FW = 1.0f;//gFilterWeights[(int)fabs((float)y - Y)];

		const float4 ColorXyza = tex2D(gTexFrameBlurXyza, X + 0.5f, y + 0.5f);

		Sum		+= FW * CColorXyz(ColorXyza.x, ColorXyza.y, ColorXyza.z);
		SumW	+= FW;
	}

	Sum /= SumW;

	const float4 ColorXYZA = make_float4(Sum.c[0], Sum.c[1], Sum.c[2], 0.0f);
	surf2Dwrite(ColorXYZA, gSurfFrameEstimateXyza, X * sizeof(float4), Y);
}

void BlurImageXyz(CScene* pScene, CScene* pDevScene)
{
	const dim3 KernelBlock(KRNL_BLUR_BLOCK_W, KRNL_BLUR_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));

	KrnlBlurXyzH<<<KernelGrid, KernelBlock>>>();
	HandleCudaError(cudaGetLastError());

	KrnlBlurXyzV<<<KernelGrid, KernelBlock>>>();
	HandleCudaError(cudaGetLastError());
}