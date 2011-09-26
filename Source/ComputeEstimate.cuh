#pragma once

#include "Geometry.h"
#include "Variance.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

KERNEL void KrnlEstimate(CScene* pScene, CColorXyz* pEstFrameXyz, CColorXyz* pAccEstXyz, CColorXyz* pEstXyz, CColorRgbaLdr* pPixels, float N, CVariance* pVariance)
{
	const int X 	= (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global Y
	const int Y		= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global X
	const int PID	= (Y * pScene->m_Camera.m_Film.GetWidth()) + X;								// Get pixel ID	

	// Exit if beyond image boundaries
	if (X >= pScene->m_Camera.m_Film.GetWidth() || Y >= pScene->m_Camera.m_Film.GetHeight())
		return;

	pAccEstXyz[PID] += pEstFrameXyz[PID];

	pEstXyz[PID] = pAccEstXyz[PID] / (float)__max(1.0f, N);
	
	float InvGamma = 1.0f / 2.2f;

	CColorRgbHdr RgbHdr, RgbHdr2;
	
	RgbHdr.FromXYZ(pEstXyz[PID].c[0], pEstXyz[PID].c[1], pEstXyz[PID].c[2]);

//	RgbHdr.FromXYZ(pEstFrameXyz[PID].c[0], pEstFrameXyz[PID].c[1], pEstFrameXyz[PID].c[2]);

	// Tone mapping
	RgbHdr.r = Clamp(1.0f - expf(-(RgbHdr.r / pScene->m_Camera.m_Film.m_Exposure)), 0.0, 1.0f);
	RgbHdr.g = Clamp(1.0f - expf(-(RgbHdr.g / pScene->m_Camera.m_Film.m_Exposure)), 0.0, 1.0f);
	RgbHdr.b = Clamp(1.0f - expf(-(RgbHdr.b / pScene->m_Camera.m_Film.m_Exposure)), 0.0, 1.0f);

	// Add sample variance
	pVariance->Push(RgbHdr.g, PID);

	// Get variance
	const float Variance = pVariance->GetVariance(PID);

	// Uncomment to show variance
	pPixels[PID].r = (unsigned char)Clamp((255.0f * powf(Variance, InvGamma)), 0.0f, 255.0f);
	pPixels[PID].g = (unsigned char)Clamp((255.0f * powf(Variance, InvGamma)), 0.0f, 255.0f);
	pPixels[PID].b = (unsigned char)Clamp((255.0f * powf(Variance, InvGamma)), 0.0f, 255.0f);
	
	// Uncomment to show normal image
// 	pPixels[PID].r = (unsigned char)Clamp((255.0f * powf(pVariance->Variance(PID), InvGamma)), 0.0f, 255.0f);
// 	pPixels[PID].g = (unsigned char)Clamp((255.0f * powf(pVariance->Variance(PID), InvGamma)), 0.0f, 255.0f);
// 	pPixels[PID].b = (unsigned char)Clamp((255.0f * powf(pVariance->Variance(PID), InvGamma)), 0.0f, 255.0f);
}

void Estimate(CScene* pScene, CScene* pDevScene, CColorXyz* pEstFrameXyz, CColorXyz* pAccEstXyz, CColorXyz* pEstXyz, CColorRgbaLdr* pPixels, float N, CVariance* pDevVariance)
{
	const dim3 KernelBlock(16, 8);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.GetWidth() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.GetHeight() / (float)KernelBlock.y));

	KrnlEstimate<<<KernelGrid, KernelBlock>>>(pDevScene, pEstFrameXyz, pAccEstXyz, pEstXyz, pPixels, N, pDevVariance); 

	
}