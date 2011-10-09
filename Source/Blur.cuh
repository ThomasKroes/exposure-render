#pragma once

#include "Geometry.h"
#include "Filter.h"
#include "Scene.h"

#define KRNL_BLUR_BLOCK_W		32
#define KRNL_BLUR_BLOCK_H		8
#define KRNL_BLUR_BLOCK_SIZE	KRNL_BLUR_BLOCK_W * KRNL_BLUR_BLOCK_H

KERNEL void KrnlBlurXyzH(CColorXyz* pImage, CColorXyz* pTempImage)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int PID	= Y * gFilmWidth + X;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	const int X0 = max((int)ceilf(X - gFilterWidth), 0);
	const int X1 = min((int)floorf(X + gFilterWidth), (int)gFilmWidth - 1);

	CColorXyz Sum;

	float FW = 1.0f, SumW = 0.0f;

	for (int x = X0; x <= X1; x++)
	{
		FW = gFilterWeights[(int)fabs((float)x - X)];

		Sum		+= FW * pImage[(Y * gFilmWidth) + x];
		SumW	+= FW;
	}

	__syncthreads();

	pTempImage[PID] = Sum / SumW;
}

KERNEL void KrnlBlurXyzV(CColorXyz* pImage, CColorXyz* pTempImage)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int PID	= Y * gFilmWidth + X;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	const int Y0 = max((int)ceilf (Y - gFilterWidth), 0);
	const int Y1 = min((int)floorf(Y + gFilterWidth), gFilmHeight - 1);

	CColorXyz Sum;

	float FW = 1.0f, SumW = 0.0f;

	for (int y = Y0; y <= Y1; y++)
	{
		FW = gFilterWeights[(int)fabs((float)y - Y)];

		Sum		+= FW * pTempImage[(y * gFilmWidth) + X];
		SumW	+= FW;
	}

	__syncthreads();

	pImage[PID]	= Sum / SumW;
}

void BlurImageXyz(CScene* pScene, CScene* pDevScene, CColorXyz* pImage, CColorXyz* pTempImage)
{
	const dim3 KernelBlock(KRNL_BLUR_BLOCK_W, KRNL_BLUR_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));

	KrnlBlurXyzH<<<KernelGrid, KernelBlock>>>(pImage, pTempImage); 
	KrnlBlurXyzV<<<KernelGrid, KernelBlock>>>(pImage, pTempImage); 
}