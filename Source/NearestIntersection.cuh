/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "CudaUtilities.h"

#define KRNL_NI_BLOCK_W		1
#define KRNL_NI_BLOCK_H		1
#define KRNL_NI_BLOCK_SIZE	KRNL_NI_BLOCK_W * KRNL_NI_BLOCK_H

#define KRNL_ZBUFFER_BLOCK_W		8
#define KRNL_ZBUFFER_BLOCK_H		8
#define KRNL_ZBUFFER_BLOCK_SIZE		KRNL_ZBUFFER_BLOCK_W * KRNL_ZBUFFER_BLOCK_H

KERNEL void KrnlNearestIntersection(CScene* pScene, float* pT)
{
	CRay Rc;
	
	const Vec2f UV(0.5f * (float)gFilmWidth, 0.5f * (float)gFilmHeight);

	pScene->m_Camera.GenerateRay(UV, Vec2f(0.0f), Rc.m_O, Rc.m_D);

	Rc.m_MinT = 0.0f;
	Rc.m_MaxT = INF_MAX;

//	NearestIntersection(Rc, pScene, *pT);
}

float NearestIntersection(CScene* pDevScene)
{
	const dim3 KernelBlock(KRNL_NI_BLOCK_W, KRNL_NI_BLOCK_H);
	const dim3 KernelGrid(1, 1);
	
	float T = 0.0f;

	float* pDevT = NULL;

	HandleCudaError(cudaMalloc(&pDevT, sizeof(float)));

	KrnlNearestIntersection<<<KernelGrid, KernelBlock>>>(pDevScene, pDevT);
	HandleCudaError(cudaGetLastError());
	cudaThreadSynchronize();

	HandleCudaError(cudaMemcpy(&T, pDevT, sizeof(float), cudaMemcpyDeviceToHost));
	HandleCudaError(cudaFree(pDevT));

	return T;
}

KERNEL void KrnlZBuffer(CScene* pScene, CCudaView* pView)
{
	const int X		= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;
	
	CRNG RNG(pView->m_RandomSeeds1.GetPtr(X, Y), pView->m_RandomSeeds2.GetPtr(X, Y));

	ColorXYZf Lv = SPEC_BLACK, Li = SPEC_BLACK;

	CRay Re;
	
	const Vec2f UV = Vec2f(X, Y) + RNG.Get2();

	float MinT = INF_MAX;

	for (int i = 0; i < 1; i++)
	{
 		pScene->m_Camera.GenerateRay(UV, RNG.Get2(), Re.m_O, Re.m_D);

		Re.m_MinT = 0.0f;
		Re.m_MaxT = INF_MAX;

		float T = 0.0f;

		if (NearestIntersection(Re, RNG, pScene, T) && T < MinT)
		{
			MinT = T;
		}
	}

//	if (MinT < INF_MAX)
		pView->m_ZBuffer.Set(MinT, X, Y);
}

void CreateZBuffer(CScene* pScene, CScene* pDevScene, CCudaView* pDevView)
{
	const dim3 KernelBlock(KRNL_ZBUFFER_BLOCK_W, KRNL_ZBUFFER_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));

	KrnlZBuffer<<<KernelGrid, KernelBlock>>>(pDevScene, pDevView);
	cudaThreadSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "ZBuffer");
}