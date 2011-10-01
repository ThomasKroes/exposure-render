
// Precompiled headers
#include "Stable.h"

#include "CudaFrameBuffers.h"
#include "CudaUtilities.h"

CCudaFrameBuffers::CCudaFrameBuffers(void) :
	m_pDevAccEstXyz(NULL),
	m_pDevEstXyz(NULL),
	m_pDevEstFrameXyz(NULL),
	m_pDevEstFrameBlurXyz(NULL),
	m_pDevEstRgbaLdr(NULL),
	m_pDevRgbLdrDisp(NULL),
	m_pDevSeeds(NULL)
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
	m_pDevRgbLdrDisp		= Other.m_pDevRgbLdrDisp;
	m_pDevSeeds				= Other.m_pDevSeeds;

	return *this;
}

void CCudaFrameBuffers::Resize(const Vec2i& Resolution)
{
	m_Resolution = Resolution;

	const int NoPixels = Resolution.x * Resolution.y;

	const int NoRandomSeeds = NoPixels * 2;

	int SizeAccEstXyz		= NoPixels * sizeof(CColorXyz);
	int SizeEstXyz			= NoPixels * sizeof(CColorXyz);
	int SizeEstFrameXyz		= NoPixels * sizeof(CColorXyz);
	int SizeEstFrameBlurXyz	= NoPixels * sizeof(CColorXyz);
	int SizeEstRgbaLdr		= NoPixels * sizeof(CColorRgbaLdr);
	int SizeRgbLdrDisp		= NoPixels * sizeof(CColorRgbLdr);
	int SizeRandomSeeds		= NoRandomSeeds * sizeof(unsigned int);

	Free();

	HandleCudaError(cudaMalloc((void**)&m_pDevSeeds, SizeRandomSeeds));
	HandleCudaError(cudaMalloc((void**)&m_pDevAccEstXyz, SizeAccEstXyz));
	HandleCudaError(cudaMalloc((void**)&m_pDevEstXyz, SizeEstXyz));
	HandleCudaError(cudaMalloc((void**)&m_pDevEstFrameXyz, SizeEstFrameXyz));
	HandleCudaError(cudaMalloc((void**)&m_pDevEstFrameBlurXyz, SizeEstFrameBlurXyz));
	HandleCudaError(cudaMalloc((void**)&m_pDevEstRgbaLdr, SizeEstRgbaLdr));
	HandleCudaError(cudaMalloc((void**)&m_pDevRgbLdrDisp, SizeRgbLdrDisp));

	// Create random seeds
	unsigned int* pSeeds = (unsigned int*)malloc(SizeRandomSeeds);

	for (int i = 0; i < NoRandomSeeds; i++)
		pSeeds[i] = qrand();

	HandleCudaError(cudaMemcpy(m_pDevSeeds, pSeeds, SizeRandomSeeds, cudaMemcpyHostToDevice));

	free(pSeeds);

	// Reset buffers to black
	Reset();

	gStatus.SetStatisticChanged("CUDA Memory", "Random Seeds", QString::number((float)SizeRandomSeeds / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("Frame Buffers", "Accumulated Estimate (XYZ color)", QString::number((float)SizeAccEstXyz / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("Frame Buffers", "Estimate (XYZ color)", QString::number((float)SizeEstXyz / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("Frame Buffers", "Frame Estimate (XYZ color)", QString::number((float)SizeEstFrameXyz / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("Frame Buffers", "Blur Frame Estimate (XYZ color)", QString::number((float)SizeEstFrameBlurXyz / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("Frame Buffers", "Estimate (RGB color)", QString::number((float)SizeEstRgbaLdr / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("Frame Buffers", "Estimate Screen (RGB color)", QString::number((float)SizeRgbLdrDisp / MB, 'f', 2), "MB");

	gStatus.SetStatisticChanged("Film", "Width SceneCopy", QString::number(m_Resolution.x), "Pixels");
	gStatus.SetStatisticChanged("Film", "Height SceneCopy", QString::number(m_Resolution.y), "Pixels");
}

void CCudaFrameBuffers::Reset(void)
{
	const int NoPixels = m_Resolution.x * m_Resolution.y;

	HandleCudaError(cudaMemset(m_pDevAccEstXyz, 0, NoPixels * sizeof(CColorXyz)));
	HandleCudaError(cudaMemset(m_pDevEstXyz, 0, NoPixels * sizeof(CColorXyz)));
	HandleCudaError(cudaMemset(m_pDevEstFrameXyz, 0, NoPixels * sizeof(CColorXyz)));
	HandleCudaError(cudaMemset(m_pDevEstFrameBlurXyz, 0, NoPixels * sizeof(CColorXyz)));
	HandleCudaError(cudaMemset(m_pDevAccEstXyz, 0, NoPixels * sizeof(CColorXyz)));
	HandleCudaError(cudaMemset(m_pDevAccEstXyz, 0, NoPixels * sizeof(CColorXyz)));
}

void CCudaFrameBuffers::Free(void)
{
	HandleCudaError(cudaFree(m_pDevAccEstXyz));
	HandleCudaError(cudaFree(m_pDevEstXyz));
	HandleCudaError(cudaFree(m_pDevEstFrameXyz));
	HandleCudaError(cudaFree(m_pDevEstFrameBlurXyz));
	HandleCudaError(cudaFree(m_pDevEstRgbaLdr));
	HandleCudaError(cudaFree(m_pDevRgbLdrDisp));
	HandleCudaError(cudaFree(m_pDevSeeds));

	m_pDevAccEstXyz			= NULL;
	m_pDevEstXyz			= NULL;
	m_pDevEstFrameXyz		= NULL;
	m_pDevEstFrameBlurXyz	= NULL;
	m_pDevEstRgbaLdr		= NULL;
	m_pDevRgbLdrDisp		= NULL;
	m_pDevSeeds				= NULL;
}