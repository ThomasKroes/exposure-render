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

#include "Transport.cuh"

KERNEL void KrnlSingleScattering(VolumeInfo* pVolumeInfo, RenderInfo* pRenderInfo, FrameBuffer* pFrameBuffer)
{
	
	const int X		= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= pRenderInfo->m_FilmWidth || Y >= pRenderInfo->m_FilmHeight)
		return;
	
	CRNG RNG(pFrameBuffer->m_RandomSeeds1.GetPtr(X, Y), pFrameBuffer->m_RandomSeeds2.GetPtr(X, Y));

	ColorRGBAuc Col = ColorRGBAuc(RNG.Get1() * 255.0f, RNG.Get1() * 255.0f, RNG.Get1() * 255.0f, RNG.Get1() * 255.0f);//255.0f, RNG.Get1() * 255.0f, RNG.Get1() * 255.0f, 150.0f);

	

	Vec2f ScreenPoint;

	ScreenPoint.x = pRenderInfo->m_Camera.m_Screen[0][0] + (pRenderInfo->m_Camera.m_InvScreen.x * (float)X);
	ScreenPoint.y = pRenderInfo->m_Camera.m_Screen[1][0] + (pRenderInfo->m_Camera.m_InvScreen.y * (float)Y);

	CRay Re;

	Re.m_O	= pRenderInfo->m_Camera.m_Pos;
	Re.m_D	= Normalize(pRenderInfo->m_Camera.m_N + (-ScreenPoint.x * pRenderInfo->m_Camera.m_U) + (-ScreenPoint.y * pRenderInfo->m_Camera.m_V));
	Re.m_MinT	= 0.0f;
	Re.m_MaxT	= 10000.0f;

	float Near = 0.0f, Far = 150000.0f;

	if (IntersectBox(Re, &Near, &Far, *pVolumeInfo))
		Col = ColorRGBAuc(255, 0, 0, 120);
	else
		Col = ColorRGBAuc(0, 255, 0, 120);

	pFrameBuffer->m_EstimateRgbaLdr.Set(Col, X, Y);


/*
	ColorXYZf Lv = SPEC_BLACK, Li = SPEC_BLACK;

	CRay Re;
	
	const Vec2f UV = Vec2f(X, Y) + RNG.Get2();

 	pScene->m_Camera.GenerateRay(UV, RNG.Get2(), Re.m_O, Re.m_D);

	Re.m_MinT = 0.0f;//pView->m_ZBuffer.Get(X, Y)-0.01f; 
	Re.m_MaxT = INF_MAX;

	Vec3f Pe, Pl;
	
	CLight* pLight = NULL;
	
	if (Re.m_MinT == INF_MAX)
		return;

	if (SampleDistanceRM(Re, RNG, Pe))
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, (Pe - Re.m_O).Length()), Li, Pl, pLight))
		{
			pView->m_FrameEstimateXyza.Set(ColorXYZAf(Lv), X, Y);
			return;
		}
		
		const float D = GetNormalizedIntensity(Pe);

		Lv += GetEmission(D);

		switch (pScene->m_ShadingType)
		{
			case 0:
			{
				Lv += UniformSampleOneLight(pScene, CVolumeShader::Brdf, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe), RNG, true);
				break;
			}
		
			case 1:
			{
				Lv += 0.5f * UniformSampleOneLight(pScene, CVolumeShader::Phase, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe), RNG, false);
				break;
			}

			case 2:
			{
				const float GradMag = GradientMagnitude(Pe) * gVolumeInfo.m_IntensityInvRange;
				const float PdfBrdf = (1.0f - __expf(-pScene->m_GradientFactor * GradMag));

				if (RNG.Get1() < PdfBrdf)
  					Lv += UniformSampleOneLight(pScene, CVolumeShader::Brdf, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe), RNG, true);
				else
					Lv += 0.5f * UniformSampleOneLight(pScene, CVolumeShader::Phase, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe), RNG, false);

				break;
			}
		}
	}
	else
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, INF_MAX), Li, Pl, pLight))
			Lv = Li;
	}

	pView->m_FrameEstimateXyza.Set(ColorXYZAf(Lv), X, Y);
	*/
}

void SingleScattering(dim3 BlockDim, dim3 GridDim, VolumeInfo* pDevVolumeInfo, RenderInfo* pDevRenderInfo, FrameBuffer* pFrameBuffer)
{
	KrnlSingleScattering<<<GridDim, BlockDim>>>(pDevVolumeInfo, pDevRenderInfo, pFrameBuffer);
	cudaThreadSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Single Scattering");
}