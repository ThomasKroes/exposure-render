#pragma once

#include "Geometry.h"
#include "Variance.h"
#include "CudaFrameBuffers.h"

#define KRNL_TM_BLOCK_W		32
#define KRNL_TM_BLOCK_H		8
#define KRNL_TM_BLOCK_SIZE	KRNL_TM_BLOCK_W * KRNL_TM_BLOCK_H

KERNEL void KrnlToneMap(CScene* pScene)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	const float4 Color = tex2D(gTexRunningEstimateXyza, X, Y) + tex2D(gTexRunningSpecularBloomXyza, X, Y);

	CColorRgbHdr RgbHdr;

	RgbHdr.FromXYZ(Color.x, Color.y, Color.z);

	RgbHdr.r = Clamp(1.0f - expf(-(RgbHdr.r * gInvExposure)), 0.0, 1.0f);
	RgbHdr.g = Clamp(1.0f - expf(-(RgbHdr.g * gInvExposure)), 0.0, 1.0f);
	RgbHdr.b = Clamp(1.0f - expf(-(RgbHdr.b * gInvExposure)), 0.0, 1.0f);

	surf2Dwrite(make_uchar4(RgbHdr.r * 255.0f, RgbHdr.g * 255.0f, RgbHdr.b * 255.0f, 0.0f), gSurfRunningEstimateRgba, X * sizeof(uchar4), Y);
}

void ToneMap(CScene* pScene, CScene* pDevScene, CCudaFrameBuffers& CudaFrameBuffers)
{
	const dim3 KernelBlock(KRNL_TM_BLOCK_W, KRNL_TM_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.GetWidth() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.GetHeight() / (float)KernelBlock.y));

	KrnlToneMap<<<KernelGrid, KernelBlock>>>(pDevScene);
	HandleCudaError(cudaGetLastError());
}