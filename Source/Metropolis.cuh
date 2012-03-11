
#include "SingleScattering.cuh"

#include <thrust/reduce.h>

#define KRNL_ADV_MLT_BLOCK_W		16
#define KRNL_ADV_MLT_BLOCK_H		8
#define KRNL_ADV_MLT_BLOCK_SIZE		KRNL_ADV_MLT_BLOCK_W * KRNL_ADV_MLT_BLOCK_H

/**/
KERNEL void KrnlAdvanceMetropolis(FrameBuffer* pFrameBuffer)
{
	const int X		= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= pFrameBuffer->Resolution[0] || Y >= pFrameBuffer->Resolution[1])
		return;
	
	CRNG RNG(pFrameBuffer->CudaRandomSeeds1.GetPtr(X, Y), pFrameBuffer->CudaRandomSeeds2.GetPtr(X, Y));

	const float pLarge = 0.5f;
	
	bool Large = RNG.Get1() < pLarge;

	MetroSample& Sample = pFrameBuffer->CudaMetroSamples(X, Y);

	if (gScattering.NoIterations == 0)
		Sample.LargeStep(RNG);

	MetroSample NewSample = Large ? MetroSample(RNG) : Sample.Mutate(RNG);

	Ray Rc;

	SampleCamera(Rc, NewSample.CameraSample);

	ColorXYZf Lv = SPEC_BLACK, Li = SPEC_BLACK, Throughput = ColorXYZf(1.0f);

	const ScatterEvent SE = SampleRay(Rc, RNG);
	
	if (SE.Valid && SE.Type == ScatterEvent::ErVolume)
	{
		Lv += UniformSampleOneLightVolume(SE, RNG, Sample.LightingSample);
	}

	/*
	if (SE.Valid && SE.Type == ScatterEvent::Light)
	{
		Lv += SE.Le;
	}

	if (SE.Valid && SE.Type == ScatterEvent::Reflector)
	{
		CVolumeShader Shader(CVolumeShader::Brdf, SE.N, SE.Wo, ColorXYZf(0.2f), ColorXYZf(0.5f), 5.0f, 10.0f);
		Lv += UniformSampleOneLight(SE, RNG, Shader, Sample.LightingSample);
	}
	*/

	ColorXYZAf NewL;

	NewL[0] = Lv.GetX();//, Lv.GetY(), Lv.GetZ(), SE.Valid >= 0 ? 1.0f : 0.0f);
	NewL[1] = Lv.GetY();
	NewL[2] = Lv.GetZ();
	NewL[3] = SE.Valid ? 1.0f : 0.0f;

	const int OldXY[] = { Sample.CameraSample.FilmUV[0] * (float)pFrameBuffer->Resolution[0], Sample.CameraSample.FilmUV[1] * (float)pFrameBuffer->Resolution[1] };
	const int NewXY[] = { NewSample.CameraSample.FilmUV[0] * (float)pFrameBuffer->Resolution[0], NewSample.CameraSample.FilmUV[1] * (float)pFrameBuffer->Resolution[1] };

	if (Large)
	{
		pFrameBuffer->CudaPixelLuminance(NewXY[0], NewXY[1]) += NewL.Y();
		pFrameBuffer->CudaNoIterations(NewXY[0], NewXY[1])++;
	}

	const float AccProb = 0.5f;//gScattering.NoIterations == 0 ? 1.0f : __min(1, NewL.Y() / Sample.OldL.Y());

	if (Sample.OldL.Y() > 0)
	{
		pFrameBuffer->CudaFrameEstimateTestXyza(OldXY[0], OldXY[1]) = Sample.OldL * (1 / Sample.OldL.Y()) * (1 - AccProb);
	}

	if (NewL.Y() > 0)
	{
		pFrameBuffer->CudaFrameEstimateTestXyza(NewXY[0], NewXY[1]) = NewL * (1 / NewL.Y()) * AccProb;
	}

	if (RNG.Get1() < AccProb)
	{
		Sample = NewSample;
		Sample.OldL = NewL;
	}
}

void AdvanceMetropolis(FrameBuffer* pFrameBuffer)
{
	const dim3 BlockDim(KRNL_ADV_MLT_BLOCK_W, KRNL_ADV_MLT_BLOCK_H);
	const dim3 GridDim((float)METRO_SIZE / (float)BlockDim.x, (float)METRO_SIZE / (float)BlockDim.y);

	FB.CudaFrameEstimateTestXyza.Reset();

	KrnlAdvanceMetropolis<<<GridDim, BlockDim>>>(pFrameBuffer);
	cudaThreadSynchronize();
}

#define KRNL_ESTIMATE_BLOCK_W		16
#define KRNL_ESTIMATE_BLOCK_H		8
#define KRNL_ESTIMATE_BLOCK_SIZE	KRNL_ESTIMATE_BLOCK_W * KRNL_ESTIMATE_BLOCK_H

KERNEL void KrnlComputeEstimateMetropolis(FrameBuffer* pFrameBuffer, float AverageLuminance, float SPP)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= pFrameBuffer->Resolution[0] || Y >= pFrameBuffer->Resolution[1])
		return;

	pFrameBuffer->CudaFrameEstimateXyza(X, Y) += pFrameBuffer->CudaFrameEstimateTestXyza(X, Y);

	pFrameBuffer->CudaRunningEstimateXyza(X, Y) = pFrameBuffer->CudaFrameEstimateXyza(X, Y) * AverageLuminance / SPP / 4.0f;
}

void ComputeEstimateMetropolis(FrameBuffer* pFrameBuffer, int Width, int Height)
{
	const dim3 BlockDim(KRNL_ESTIMATE_BLOCK_W, KRNL_ESTIMATE_BLOCK_H);
	const dim3 GridDim((int)ceilf((float)Width / (float)BlockDim.x), (int)ceilf((float)Height / (float)BlockDim.y));

	thrust::device_ptr<int> DevPtrSumNoMetro(FB.CudaNoIterations.GetPtr()); 
	int SumNoMetro = thrust::reduce(DevPtrSumNoMetro, DevPtrSumNoMetro + Width * Height);

	thrust::device_ptr<float> DevPtrSumPixelLuminance(FB.CudaPixelLuminance.GetPtr()); 
	float SumLuminance = thrust::reduce(DevPtrSumPixelLuminance, DevPtrSumPixelLuminance + Width * Height);
	
	float AverageLuminance = SumLuminance / (float)SumNoMetro;
	float SPP = (float)SumNoMetro / (Width * Height);

	KrnlComputeEstimateMetropolis<<<GridDim, BlockDim>>>(pFrameBuffer, AverageLuminance, SPP);
	cudaThreadSynchronize();
}
