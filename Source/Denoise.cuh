
#include "Scene.h"

#include <cuda_runtime.h>
#include <cutil.h>

DEV float lerpf(float a, float b, float c){
	return a + (b - a) * c;
}

DEV float vecLen(float4 a, float4 b)
{
    return ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) + (b.z - a.z) * (b.z - a.z));
}

KERNEL void KNN(CScene* pScene, CColorRgbaLdr* pOut)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    //Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

	int ID = pScene->m_Camera.m_Film.m_Resolution.GetResX() * (pScene->m_Camera.m_Film.m_Resolution.GetResY() - iy) + ix;

    if(pScene->m_DenoiseParams.m_Enabled && ix < pScene->m_Camera.m_Film.m_Resolution.GetResX() && iy < pScene->m_Camera.m_Film.m_Resolution.GetResY())
	{
        //Normalized counter for the weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float3 clr = {0, 0, 0};
        //Center of the KNN window
        float4 clr00 = tex2D(gTexEstimateRgbLdr, x, y);

        //Cycle through KNN window, surrounding (x, y) texel
        for(float i = -pScene->m_DenoiseParams.m_WindowRadius; i <= pScene->m_DenoiseParams.m_WindowRadius; i++)
            for(float j = -pScene->m_DenoiseParams.m_WindowRadius; j <= pScene->m_DenoiseParams.m_WindowRadius; j++)
            {
                float4     clrIJ = tex2D(gTexEstimateRgbLdr, x + j, y + i);
                float distanceIJ = vecLen(clr00, clrIJ);

                //Derive final weight from color distance
                float   weightIJ = __expf( - (distanceIJ * pScene->m_DenoiseParams.m_Noise + (i * i + j * j) * pScene->m_DenoiseParams.m_InvWindowArea) );

                //Accumulate (x + j, y + i) texel color with computed weight
                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights     += weightIJ;

                //Update weight counter, if KNN weight for current window texel
                //exceeds the weight threshold
                fCount         += (weightIJ > pScene->m_DenoiseParams.m_WeightThreshold) ? pScene->m_DenoiseParams.m_InvWindowArea : 0;
            }

        //Normalize result color by sum of weights
        sumWeights = 1.0f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        //Choose LERP quotent basing on how many texels
        //within the KNN window exceeded the weight threshold
        float lerpQ = (fCount > pScene->m_DenoiseParams.m_LerpThreshold) ? pScene->m_DenoiseParams.m_LerpC : 1.0f - pScene->m_DenoiseParams.m_LerpC;

        // Write final result to global memory
        clr.x = lerpf(clr.x, clr00.x, lerpQ);
        clr.y = lerpf(clr.y, clr00.y, lerpQ);
        clr.z = lerpf(clr.z, clr00.z, lerpQ);

		pOut[ID].r = 255 * clr.x;
		pOut[ID].g = 255 * clr.y;
		pOut[ID].b = 255 * clr.z;
    }
	else
	{
		float4 clr00 = tex2D(gTexEstimateRgbLdr, x, y);

		pOut[ID].r = 255 * clr00.x;
		pOut[ID].g = 255 * clr00.y;
		pOut[ID].b = 255 * clr00.z;
	}

	// Add edge around image
// 	if (ix == 0 || ix == (pScene->m_Camera.m_Film.m_Resolution.GetResX() - 1) || iy == 0 || iy == (pScene->m_Camera.m_Film.m_Resolution.GetResY() - 1))
// 	{
// 		pOut[ID].r = 0;
// 		pOut[ID].g = 0;
// 		pOut[ID].b = 0;
// 	}
}

void Denoise(CScene* pScene, CScene* pDevScene, CColorRgbaLdr* pIn, CColorRgbaLdr* pOut)
{
	const dim3 KernelBlock(16, 8);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));

	// De-noise the image
//	KrnlDenoise<<<KernelGrid, KernelBlock>>>(pDevScene, pIn, pOut);
	KNN<<<KernelGrid, KernelBlock>>>(pDevScene, pOut);
}