
#include <FidelityRender\Denoise.cuh>
#include <FidelityRender\CudaWrapper.h>

#define ONE_OVER_255			0.003921568f

//Texture reference and channel descriptor for image texture
texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();

// Obtains a pixel value (interpolated)
HOD CColorRgbLdr Evaluate(const Vec2f& UV, const float& Width, const float& Height, CColorRgbLdr* pIn)
{
	const float U = UV.x;//Clamp(UV.x, 0.0f, 1.0f);
	const float V = UV.y;//Clamp(UV.y, 0.0f, 1.0f);

	if (U < 0 || U >= Width || V < 0 || V >= Height)
		return CColorRgbLdr(0.0f, 0.0f, 0.0f);

	const int X0 = floorf(UV.x);
	const int X1 = ceilf(UV.x);
	const int Y0 = floorf(UV.y);
	const int Y1 = ceilf(UV.y);

	const int PID10 = Y1 * Width + X0;
	const int PID11 = Y1 * Width + X1;
	const int PID00 = Y0 * Width + X0;
	const int PID01 = Y0 * Width + X1;

	const CColorRgbLdr R00 = pIn[PID00];
	const CColorRgbLdr R01 = pIn[PID01];
	const CColorRgbLdr R10 = pIn[PID10];
	const CColorRgbLdr R11 = pIn[PID11];

	const float x = U - X0;//(U * Width) - floor(U * Width);
	const float y = V - Y0;//(V * Height) - floor(V * Height);

	return (R00 * (1.0f - x) * (1.0f - y)) + (R10 * (1.0f - x) * y) + (R01 * x * (1.0f - y)) + (R11 * x * y);
}

HOD float Length(CColorRgbHdr& A, CColorRgbHdr& B)
{
	return (B.r - A.r) * (B.r - A.r) + (B.g - A.g) * (B.g - A.g) + (B.b - A.b) * (B.b - A.b);
}

__device__ float lerpf(float a, float b, float c){
	return a + (b - a) * c;
}

KERNEL void KrnlDenoise(CCudaScene* pScene, CColorRgbLdr* pIn, CColorRgbLdr* pOut)
{
	const int X 	= (blockIdx.x * blockDim.x) + threadIdx.x;					// Get global y
	const int Y		= (blockIdx.y * blockDim.y) + threadIdx.y;					// Get global x
	const int PID	= (Y * pScene->m_Camera.m_Film.m_Resolution.m_Width) + X;	// Get pixel ID	

	// Exit if beyond image boundaries
	if (X >= pScene->m_RenderParams.m_KernelDim.m_Width || Y >= pScene->m_RenderParams.m_KernelDim.m_Height)
		return;

// 	pOut[PID] = pIn[PID];//Evaluate(Vec2f(X, Y), imageW, imageH, pIn);
// 	return;

	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	//Add half of a texel to always address exact texel centers
	const float x = (float)ix + 0.5f;
	const float y = (float)iy + 0.5f;

	if (ix < pScene->m_Camera.m_Film.m_Resolution.m_Width && iy < pScene->m_Camera.m_Film.m_Resolution.m_Height)
	{
		//Normalized counter for the weight threshold
		float fCount = 0;
		//Total sum of pixel weights
		float sumWeights = 0;
		//Result accumulator

		CColorRgbHdr Col(0.0f, 0.0f, 0.0f);

		//Center of the KNN window
		//		float4 clr00 = tex2D(texImage, x, y);
		CColorRgbLdr clr00 = Evaluate(Vec2f(x, y), pScene->m_Camera.m_Film.m_Resolution.m_Width, pScene->m_Camera.m_Film.m_Resolution.m_Height, pIn);

		CColorRgbHdr Col00(ONE_OVER_255 * clr00.r, ONE_OVER_255 * clr00.g, ONE_OVER_255 * clr00.b);

		//Cycle through KNN window, surrounding (x, y) texel
		for (float i = -pScene->m_RenderParams.m_Denoise.m_WindowRadius; i <= pScene->m_RenderParams.m_Denoise.m_WindowRadius; i++)
		{
			for (float j = -pScene->m_RenderParams.m_Denoise.m_WindowRadius; j <= pScene->m_RenderParams.m_Denoise.m_WindowRadius; j++)
			{
				//				float4     clrIJ = tex2D(texImage, x + j, y + i);
				CColorRgbLdr clrIJ = Evaluate(Vec2f(x + j, y + i), pScene->m_Camera.m_Film.m_Resolution.m_Width, pScene->m_Camera.m_Film.m_Resolution.m_Height, pIn);

				CColorRgbHdr ColIJ(ONE_OVER_255 * clrIJ.r, ONE_OVER_255 * clrIJ.g, ONE_OVER_255 * clrIJ.b);

				//				float distanceIJ = vecLen(clr00, clrIJ);
				float distanceIJ = Length(Col00, ColIJ);

				//Derive final weight from color distance
				float   weightIJ = __expf( - (distanceIJ * pScene->m_RenderParams.m_Denoise.m_Noise + (i * i + j * j) * pScene->m_RenderParams.m_Denoise.m_InvWindowArea) );

				//Accumulate (x + j, y + i) texel color with computed weight
				Col.r += ColIJ.r * weightIJ;
				Col.g += ColIJ.g * weightIJ;
				Col.b += ColIJ.b * weightIJ;

				//Sum of weights for color normalization to [0..1] range
				sumWeights     += weightIJ;

				//Update weight counter, if KNN weight for current window texel
				//exceeds the weight threshold
				fCount         += (weightIJ > pScene->m_RenderParams.m_Denoise.m_WeightThreshold) ? pScene->m_RenderParams.m_Denoise.m_InvWindowArea : 0;
			}
		}

		//Normalize result color by sum of weights
		sumWeights = 1.0f / sumWeights;
		Col.r *= sumWeights;
		Col.g *= sumWeights;
		Col.b *= sumWeights;

		//Choose LERP quotent basing on how many texels
		//within the KNN window exceeded the weight threshold
		float lerpQ = (fCount > pScene->m_RenderParams.m_Denoise.m_LerpThreshold) ? pScene->m_RenderParams.m_Denoise.m_LerpC : 1.0f - pScene->m_RenderParams.m_Denoise.m_LerpC;

		//Write Lerp result to global memory
		Col.r = (255.0f * lerpf(Col.r, Col00.r, lerpQ));
		Col.g = (255.0f * lerpf(Col.g, Col00.g, lerpQ));
		Col.b = (255.0f * lerpf(Col.b, Col00.b, lerpQ));

		int ID = pScene->m_Camera.m_Film.m_Resolution.m_Width * iy + ix;

		pOut[ID] = CColorRgbLdr((unsigned char)Col.r, (unsigned char)Col.g, (unsigned char)Col.b);
	};
}

void Denoise(CCudaScene& Scene, CCudaScene* pDevCudaScene, CStatistics& Statistics)
{
	CCudaTimer Timer;

	const dim3 KernelBlock(Scene.m_RenderParams.m_BlockDim.m_Width, Scene.m_RenderParams.m_BlockDim.m_Height);
	const dim3 KernelGrid((int)ceilf((float)800.0f / (float)KernelBlock.x), (int)ceilf((float)Scene.m_RenderParams.m_KernelDim.m_Height / (float)KernelBlock.y));

	// De-noise the image
	KrnlDenoise<<<KernelGrid, KernelBlock>>>(pDevCudaScene, Scene.m_FrameBuffer.m_pEstRgbLdr, Scene.m_FrameBuffer.m_pDispRgbLdr);
	HANDLE_CUDA_ERROR(cudaGetLastError(), "'KrnlDenoise' failed");

	Statistics.m_KernelEvents.AddDuration("Denoise", Timer.StopTimer());
}