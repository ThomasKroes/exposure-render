/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Utilities.cuh"
#include "CudaUtilities.h"

#define KRNL_DENOISE_BLOCK_W		16
#define KRNL_DENOISE_BLOCK_H		8
#define KRNL_DENOISE_BLOCK_SIZE		KRNL_DENOISE_BLOCK_W * KRNL_DENOISE_BLOCK_H

DEV float lerpf(float a, float b, float c){
	return a + (b - a) * c;
}

DEV float vecLen(float4 a, float4 b)
{
    return ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) + (b.z - a.z) * (b.z - a.z));
}

KERNEL void KrnlDenoise(RenderInfo* pRenderInfo, FrameBuffer* pFrameBuffer)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

//	const ColorRGBuc RGBA = pFrameBuffer->m_EstimateRgbLdr.Get(X, Y);
//	pFrameBuffer->m_DisplayEstimateRgbLdr.Set(ColorRGBuc(RGBA.GetR(), RGBA.GetG(), RGBA.GetB()), X, Y);

	/*
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= gRenderInfo.m_FilmWidth || Y >= gRenderInfo.m_FilmHeight)
		return;

    const float x = (float)X + 0.5f;
    const float y = (float)Y + 0.5f;

	const float4 clr00 = tex2D(gTexRunningEstimateRgba, x, y);
	
	if (gRenderInfo.m_Denoise.m_Enabled && gRenderInfo.m_Denoise.m_LerpC > 0.0f && gRenderInfo.m_Denoise.m_LerpC < 1.0f)
	{
        float			fCount		= 0;
        float			SumWeights	= 0;
        float3			clr			= { 0, 0, 0 };
        		
        for (float i = -gRenderInfo.m_Denoise.m_WindowRadius; i <= gRenderInfo.m_Denoise.m_WindowRadius; i++)
		{
            for (float j = -gRenderInfo.m_Denoise.m_WindowRadius; j <= gRenderInfo.m_Denoise.m_WindowRadius; j++)
            {
                const float4 clrIJ = tex2D(gTexRunningEstimateRgba, x + j, y + i);
                const float distanceIJ = vecLen(clr00, clrIJ);

                const float weightIJ = __expf(-(distanceIJ * gRenderInfo.m_Denoise.m_Noise + (i * i + j * j) * gRenderInfo.m_Denoise.m_InvWindowArea));

                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                SumWeights += weightIJ;

                fCount += (weightIJ > gRenderInfo.m_Denoise.m_WeightThreshold) ? gRenderInfo.m_Denoise.m_InvWindowArea : 0;
            }
		}
		
		SumWeights = 1.0f / SumWeights;

		clr.x *= SumWeights;
		clr.y *= SumWeights;
		clr.z *= SumWeights;

		const float LerpQ = (fCount > gRenderInfo.m_Denoise.m_LerpThreshold) ? gRenderInfo.m_Denoise.m_LerpC : 1.0f - gRenderInfo.m_Denoise.m_LerpC;

		clr.x = lerpf(clr.x, clr00.x, LerpQ);
		clr.y = lerpf(clr.y, clr00.y, LerpQ);
		clr.z = lerpf(clr.z, clr00.z, LerpQ);

		pView->m_DisplayEstimateRgbLdr.Set(ColorRGBuc(255 * clr.x, 255 * clr.y, 255 * clr.z), X, Y);
    }
	else
	{
		const ColorRGBAuc RGBA = pView->m_EstimateRgbaLdr.Get(X, Y);
		pView->m_DisplayEstimateRgbLdr.Set(ColorRGBuc(RGBA.GetR(), RGBA.GetG(), RGBA.GetB()), X, Y);
	}
	*/
}

void Denoise(RenderInfo* pDevRenderInfo, FrameBuffer* pFrameBuffer, int Width, int Height)
{
	const dim3 BlockDim(KRNL_DENOISE_BLOCK_W, KRNL_DENOISE_BLOCK_H);
	const dim3 GridDim((int)ceilf((float)Width / (float)BlockDim.x), (int)ceilf((float)Height / (float)BlockDim.y));

	KrnlDenoise<<<GridDim, BlockDim>>>(pDevRenderInfo, pFrameBuffer);
	cudaThreadSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Noise Reduction");
}