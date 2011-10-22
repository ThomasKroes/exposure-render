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

#include "Geometry.h"
#include "Variance.h"

#define KRNL_TM_BLOCK_W		8
#define KRNL_TM_BLOCK_H		8
#define KRNL_TM_BLOCK_SIZE	KRNL_TM_BLOCK_W * KRNL_TM_BLOCK_H

/*
HOD void FromXYZ(float x, float y, float z)
{
	const float rWeight[3] = { 3.240479f, -1.537150f, -0.498535f };
	const float gWeight[3] = {-0.969256f,  1.875991f,  0.041556f };
	const float bWeight[3] = { 0.055648f, -0.204043f,  1.057311f };

	float R, G, B;

	R =	rWeight[0] * x +
		rWeight[1] * y +
		rWeight[2] * z;

	G =	gWeight[0] * x +
		gWeight[1] * y +
		gWeight[2] * z;

	B =	bWeight[0] * x +
		bWeight[1] * y +
		bWeight[2] * z;

	clamp2(R, 0.0f, 1.0f);
	clamp2(G, 0.0f, 1.0f);
	clamp2(B, 0.0f, 1.0f);

	r = (unsigned char)(R * 255.0f);
	g = (unsigned char)(G * 255.0f);
	b = (unsigned char)(B * 255.0f);
	a = 255;
}
*/

KERNEL void KrnlToneMap(CCudaView* pView)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	const ColorXYZAf Color = pView->m_RunningEstimateXyza.Get(X, Y);

	ColorRGBf RgbHdr;

	RgbHdr.FromXYZ(Color.GetX(), Color.GetY(), Color.GetZ());

	RgbHdr.SetR(Clamp(1.0f - expf(-(RgbHdr.GetR() * gInvExposure)), 0.0, 1.0f));
	RgbHdr.SetG(Clamp(1.0f - expf(-(RgbHdr.GetG() * gInvExposure)), 0.0, 1.0f));
	RgbHdr.SetB(Clamp(1.0f - expf(-(RgbHdr.GetB() * gInvExposure)), 0.0, 1.0f));

	ColorRGBAuc RGBA;

	RGBA.FromRGBAf(RgbHdr[0], RgbHdr[1], RgbHdr[2], 0.0f);

	pView->m_EstimateRgbaLdr.Set(RGBA, X, Y);
}

void ToneMap(CScene* pScene, CScene* pDevScene, CCudaView* pDevView)
{
	const dim3 KernelBlock(KRNL_TM_BLOCK_W, KRNL_TM_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.GetWidth() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.GetHeight() / (float)KernelBlock.y));

	KrnlToneMap<<<KernelGrid, KernelBlock>>>(pDevView);
	cudaThreadSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Tone Map");
}