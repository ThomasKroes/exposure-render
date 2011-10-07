
// Precompiled headers
#include "Stable.h"

#include "CudaFrameBuffers.h"
#include "CudaUtilities.h"

CCudaFrameBuffers::CCudaFrameBuffers(void) :
	m_Resolution(),
	m_pDevAccEstXyz(NULL),
	m_pDevEstXyz(NULL),
	m_pDevEstFrameXyz(NULL),
	m_pDevEstFrameBlurXyz(NULL),
	m_pDevSpecularBloom(NULL),
	m_pDevEstRgbaLdr(NULL),
	m_pDevRgbLdrDisp(NULL),
	m_pDevSeeds(NULL),
	m_pNoEstimates(NULL)
{
}

CCudaFrameBuffers::~CCudaFrameBuffers(void)
{
	Free();
}

CCudaFrameBuffers::CCudaFrameBuffers(const CCudaFrameBuffers& Other)
{
	*this = Other;
}

CCudaFrameBuffers& CCudaFrameBuffers::operator=(const CCudaFrameBuffers& Other)
{
	m_Resolution			= Other.m_Resolution;
	m_pDevAccEstXyz			= Other.m_pDevAccEstXyz;
	m_pDevEstXyz			= Other.m_pDevEstXyz;
	m_pDevEstFrameXyz		= Other.m_pDevEstFrameXyz;
	m_pDevEstFrameBlurXyz	= Other.m_pDevEstFrameBlurXyz;
	m_pDevEstRgbaLdr		= Other.m_pDevEstRgbaLdr;
	m_pDevSpecularBloom		= Other.m_pDevSpecularBloom;
	m_pDevRgbLdrDisp		= Other.m_pDevRgbLdrDisp;
	m_pDevSeeds				= Other.m_pDevSeeds;
	m_pNoEstimates			= Other.m_pNoEstimates;

	return *this;
}

void CCudaFrameBuffers::Resize(const Vec2i& Resolution)
{
	m_Resolution = Resolution;

	Free();

	long NoPixels				= Resolution.x * Resolution.y;
	long NoRandomSeeds			= NoPixels * 2;
	long SizeAccEstXyz			= NoPixels * sizeof(CColorXyz);
	long SizeEstXyz				= NoPixels * sizeof(CColorXyz);
	long SizeEstFrameXyz		= NoPixels * sizeof(CColorXyz);
	long SizeEstFrameBlurXyz	= NoPixels * sizeof(CColorXyz);
	long SizeSpecularBloom		= NoPixels * sizeof(CColorXyz);
	long SizeEstRgbaLdr			= NoPixels * sizeof(CColorRgbaLdr);
	long SizeRgbLdrDisp			= NoPixels * sizeof(CColorRgbLdr);
	long SizeRandomSeeds		= NoRandomSeeds * sizeof(int);
	long SizeNoEstimates		= NoPixels * sizeof(int);
	long Total					= SizeAccEstXyz + SizeEstXyz + SizeEstFrameXyz + SizeEstFrameBlurXyz + SizeEstRgbaLdr + SizeRgbLdrDisp + SizeRandomSeeds + SizeNoEstimates;

	HandleCudaError(cudaMalloc((void**)&m_pDevSeeds, SizeRandomSeeds));
	HandleCudaError(cudaMalloc((void**)&m_pDevAccEstXyz, SizeAccEstXyz));
	HandleCudaError(cudaMalloc((void**)&m_pDevEstXyz, SizeEstXyz));
	HandleCudaError(cudaMalloc((void**)&m_pDevEstFrameXyz, SizeEstFrameXyz));
	HandleCudaError(cudaMalloc((void**)&m_pDevEstFrameBlurXyz, SizeEstFrameBlurXyz));
	HandleCudaError(cudaMalloc((void**)&m_pDevSpecularBloom, SizeSpecularBloom));
	HandleCudaError(cudaMalloc((void**)&m_pDevEstRgbaLdr, SizeEstRgbaLdr));
	HandleCudaError(cudaMalloc((void**)&m_pDevRgbLdrDisp, SizeRgbLdrDisp));
	HandleCudaError(cudaMalloc((void**)&m_pNoEstimates, SizeNoEstimates));

	// Create random seeds
	int* pSeeds = (int*)malloc(SizeRandomSeeds);

	memset(pSeeds, 0, SizeRandomSeeds);

	for (int i = 0; i < NoRandomSeeds; i++)
		pSeeds[i] = qrand();

	HandleCudaError(cudaMemcpy(m_pDevSeeds, pSeeds, SizeRandomSeeds, cudaMemcpyHostToDevice));

	free(pSeeds);

	// Reset buffers to black
	Reset();

	gStatus.SetStatisticChanged("Frame Buffers", "Accumulated Estimate (XYZ color)", QString::number((float)SizeAccEstXyz / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("Frame Buffers", "Estimate (XYZ color)", QString::number((float)SizeEstXyz / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("Frame Buffers", "Frame Estimate (XYZ color)", QString::number((float)SizeEstFrameXyz / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("Frame Buffers", "Blur Frame Estimate (XYZ color)", QString::number((float)SizeEstFrameBlurXyz / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("Frame Buffers", "Specular Bloom (XYZ color)", QString::number((float)SizeSpecularBloom / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("Frame Buffers", "Estimate (RGBA color)", QString::number((float)SizeEstRgbaLdr / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("Frame Buffers", "Estimate Screen (RGB color)", QString::number((float)SizeRgbLdrDisp / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("Frame Buffers", "Random Seeds", QString::number((float)SizeRandomSeeds / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("Frame Buffers", "No. Estimates", QString::number((float)SizeRandomSeeds / MB, 'f', 2), "MB");

	gStatus.SetStatisticChanged("Film", "Width SceneCopy", QString::number(m_Resolution.x), "Pixels");
	gStatus.SetStatisticChanged("Film", "Height SceneCopy", QString::number(m_Resolution.y), "Pixels");

	gStatus.SetStatisticChanged("CUDA Memory", "Frame Buffers", QString::number((float)Total / MB, 'f', 2), "MB");
}

void CCudaFrameBuffers::Reset(void)
{
	const int NoPixels = m_Resolution.x * m_Resolution.y;

	HandleCudaError(cudaMemset(m_pDevAccEstXyz, 0, NoPixels * sizeof(CColorXyz)));
	HandleCudaError(cudaMemset(m_pDevEstXyz, 0, NoPixels * sizeof(CColorXyz)));
	HandleCudaError(cudaMemset(m_pDevEstFrameXyz, 0, NoPixels * sizeof(CColorXyz)));
	HandleCudaError(cudaMemset(m_pDevEstFrameBlurXyz, 0, NoPixels * sizeof(CColorXyz)));
	HandleCudaError(cudaMemset(m_pDevSpecularBloom, 0, NoPixels * sizeof(CColorXyz)));
	HandleCudaError(cudaMemset(m_pDevAccEstXyz, 0, NoPixels * sizeof(CColorXyz)));
	HandleCudaError(cudaMemset(m_pDevAccEstXyz, 0, NoPixels * sizeof(CColorXyz)));
	HandleCudaError(cudaMemset(m_pNoEstimates, 0, NoPixels * sizeof(int)));
}

void CCudaFrameBuffers::Free(void)
{
	HandleCudaError(cudaFree(m_pDevAccEstXyz));
	HandleCudaError(cudaFree(m_pDevEstXyz));
	HandleCudaError(cudaFree(m_pDevEstFrameXyz));
	HandleCudaError(cudaFree(m_pDevEstFrameBlurXyz));
	HandleCudaError(cudaFree(m_pDevSpecularBloom));
	HandleCudaError(cudaFree(m_pDevEstRgbaLdr));
	HandleCudaError(cudaFree(m_pDevRgbLdrDisp));
	HandleCudaError(cudaFree(m_pDevSeeds));
	HandleCudaError(cudaFree(m_pNoEstimates));

	m_pDevAccEstXyz			= NULL;
	m_pDevEstXyz			= NULL;
	m_pDevEstFrameXyz		= NULL;
	m_pDevEstFrameBlurXyz	= NULL;
	m_pDevSpecularBloom		= NULL;
	m_pDevEstRgbaLdr		= NULL;
	m_pDevRgbLdrDisp		= NULL;
	m_pDevSeeds				= NULL;
	m_pNoEstimates			= NULL;
}