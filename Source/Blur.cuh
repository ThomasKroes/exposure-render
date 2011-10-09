#pragma once

#include "Geometry.h"
#include "Filter.h"
#include "Scene.h"
#include "CudaUtilities.h"

#define KRNL_BLUR_BLOCK_W		32
#define KRNL_BLUR_BLOCK_H		8
#define KRNL_BLUR_BLOCK_SIZE	KRNL_BLUR_BLOCK_W * KRNL_BLUR_BLOCK_H

KERNEL void KrnlBlurXyzH(void)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	const int X0 = max((int)ceilf(X - gFilterWidth), 0);
	const int X1 = min((int)floorf(X + gFilterWidth), (int)gFilmWidth - 1);

	CColorXyz Sum;

	float FW = 1.0f, SumW = 0.0f;

	for (int x = X0; x <= X1; x++)
	{
		FW = gFilterWeights[(int)fabs((float)x - X)];

		const float4 ColorXyza = tex2D(gTexFrameEstimateXyza, x, Y);

		Sum		+= FW * CColorXyz(ColorXyza.x, ColorXyza.y, ColorXyza.z);
		SumW	+= FW;
	}

	__syncthreads();

	Sum /= SumW;

	surf2Dwrite(make_float4(Sum.c[0], Sum.c[1], Sum.c[2], 0.0f), gSurfFrameBlurXyza, X * sizeof(float4), Y);
}

KERNEL void KrnlBlurXyzV(void)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	const int Y0 = max((int)ceilf (Y - gFilterWidth), 0);
	const int Y1 = min((int)floorf(Y + gFilterWidth), gFilmHeight - 1);

	CColorXyz Sum;

	float FW = 1.0f, SumW = 0.0f;

	for (int y = Y0; y <= Y1; y++)
	{
		FW = gFilterWeights[(int)fabs((float)y - Y)];

		const float4 ColorXyza = tex2D(gTexFrameBlurXyza, X, y);

		Sum		+= FW * CColorXyz(ColorXyza.x, ColorXyza.y, ColorXyza.z);
		SumW	+= FW;
	}

	__syncthreads();

	Sum /= SumW;

	float4 ColorXYZA = make_float4(Sum.c[0], Sum.c[1], Sum.c[2], 0.0f);
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