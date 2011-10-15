#pragma once

#include "Scene.h"

#include "MonteCarlo.cuh"

#define KRNL_SP_BLOCK_W		32
#define KRNL_SP_BLOCK_H		8
#define KRNL_SP_BLOCK_SIZE	KRNL_SP_BLOCK_W * KRNL_SP_BLOCK_H

KERNEL void KrnlSpecularBloom(CCudaView* pView)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int PID	= Y * gFilmWidth + X;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	CRNG RNG(&pView->m_RandomSeeds1.m_pData[PID], &pView->m_RandomSeeds2.m_pData[PID]);

	CColorXyz Lb = SPEC_BLACK;

	float SumWeight = 0.0f;

	float BloomFilterwWidth = 2.0f;

	const int X0 = max((int)ceilf(X - BloomFilterwWidth), 0);
	const int X1 = min((int)floorf(X + BloomFilterwWidth), (int)gFilmWidth - 1);
	const int Y0 = max((int)ceilf (Y - BloomFilterwWidth), 0);
	const int Y1 = min((int)floorf(Y + BloomFilterwWidth), gFilmHeight - 1);

	float WindowWidth = 50.0f;
	float NumXY = 11.0f;
	float DXY = BloomFilterwWidth / (2.0f * BloomFilterwWidth + 1.0f);

	for (int x = X0; x < X1; x++)
	{
		for (int y = Y0; y < Y1; y++)
		{
			Vec2f UV(X + (x - X) * DXY, Y + (y - Y) * DXY);

			UV += RNG.Get2() * Vec2f(DXY, DXY);

			const float4 Le = tex2D(gTexRunningEstimateXyza, UV.x, UV.y);

			Lb += CColorXyz(Le.x, Le.y, Le.z);//expf(-3.0f * (C.Length() / Radius)) * 
			SumWeight++;
		}
	}

//		Vec2f C = Radius * ConcentricSampleDisk(RNG.Get2());

//		Vec2f UV((float)X + C.x, (float)Y + C.y);

//		if (UV.x < 0.0f || UV.x >= gFilmWidth || UV.y < 0.0f || UV.y >= gFilmHeight)
//			break;
		
//	}

	__syncthreads();

	if (SumWeight > 0.0f)
		Lb /= SumWeight;

//	float4 ColorXyza = tex2D(gTexRunningSpecularBloomXyza, X, Y) + (make_float4(Lb.c[0], Lb.c[1], Lb.c[2], 0.0f) - tex2D(gTexRunningSpecularBloomXyza, X, Y)) / (float)__max(1.0f, gNoIterations);

//	surf2Dwrite(ColorXyza, gSurfRunningSpecularBloomXyza, X * sizeof(float4), Y);
}

void SpecularBloom(CScene& Scene, CScene* pDevScene, CCudaView* pDevView)
{
	const dim3 KernelBlock(KRNL_SP_BLOCK_W, KRNL_SP_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)Scene.m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)Scene.m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));

	KrnlSpecularBloom<<<KernelGrid, KernelBlock>>>(pDevView);
	cudaThreadSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Specular Bloom");
}