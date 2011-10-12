#pragma once

#include "Geometry.h"
#include "Scene.h"
#include "CudaUtilities.h"

template<class T>
class CCudaBuffer2D
{
public:
	CCudaBuffer2D(void) :
		m_Resolution(0, 0),
		m_pData(NULL),
		m_Pitch(0)
	{
	}

	virtual ~CCudaBuffer2D(void)
	{
		Free();
	}

	CCudaBuffer2D::CCudaBuffer2D(const CCudaBuffer2D& Other)
	{
		*this = Other;
	}

	CCudaBuffer2D& CCudaBuffer2D::operator=(const CCudaBuffer2D& Other)
	{
		m_Resolution	= Other.m_Resolution;
		m_pData			= Other.m_pData;
		m_Pitch			= Other.m_Pitch;

		return *this;
	}

	void Resize(const CResolution2D& Resolution)
	{
		m_Resolution = Resolution;

		Free();

		HandleCudaError(cudaMalloc(&m_pData, m_Resolution.GetNoElements() * sizeof(T)), "Reset Cuda Buffer");
	}

	void Reset(void)
	{
//		const int Size = m_Resolution.GetNoElements() * sizeof(T);
//		HandleCudaError(cudaMemset(m_pData, 0, Size), "Reset Cuda Buffer");
	}

	void Free(void)
	{
		HandleCudaError(cudaFree(m_pData), "Free Cuda Buffer");
		m_pData = NULL;
	}

	T Read(const int& X, const int& Y)
	{
		return m_pData[Y * m_Resolution.GetResX() + X];
	}

	void Write(const int& X, const int& Y)
	{
	}

	int GetNoElements(void) const
	{
		return m_Resolution.GetNoElements();
	}

	int GetSize(void) const
	{
		return GetNoElements() * sizeof(T);
	}

	CResolution2D	m_Resolution;
	T*				m_pData;
	size_t			m_Pitch;
};

class CCudaRandomBuffer2D : public CCudaBuffer2D<int>
{
public:
	void Resize(const CResolution2D& Resolution)
	{
		CCudaBuffer2D::Resize(Resolution);

		// Create random seeds
		int* pSeeds = (int*)malloc(GetSize());

		memset(pSeeds, 0, GetSize());

		for (int i = 0; i < GetNoElements(); i++)
			pSeeds[i] = rand();

		HandleCudaError(cudaMemcpy(m_pData, pSeeds, GetSize(), cudaMemcpyHostToDevice));

		free(pSeeds);
	}
};

class CCudaView
{
public:
	CCudaView(void) :
		m_Resolution(0, 0),
		m_RunningEstimateXyza(),
		m_FrameEstimateXyza(),
		m_FrameBlurXyza(),
		m_RunningSpecularBloom(),
		m_EstimateRgbaLdr(),
		m_DisplayEstimateRgbLdr(),
		m_RandomSeeds1(),
		m_RandomSeeds2(),
		m_NoEstimates()
	{
	}

	~CCudaView(void)
	{
		Free();
	}

	CCudaView::CCudaView(const CCudaView& Other)
	{
		*this = Other;
	}

	CCudaView& CCudaView::operator=(const CCudaView& Other)
	{
		m_Resolution				= Other.m_Resolution;
		m_RunningEstimateXyza		= Other.m_RunningEstimateXyza;
		m_FrameEstimateXyza			= Other.m_FrameEstimateXyza;
		m_FrameBlurXyza				= Other.m_FrameBlurXyza;
		m_RunningSpecularBloom		= Other.m_RunningSpecularBloom;
		m_EstimateRgbaLdr			= Other.m_EstimateRgbaLdr;
		m_DisplayEstimateRgbLdr		= Other.m_DisplayEstimateRgbLdr;
		m_RandomSeeds1				= Other.m_RandomSeeds1;
		m_RandomSeeds2				= Other.m_RandomSeeds2;
		m_NoEstimates				= Other.m_NoEstimates;

		return *this;
	}

	void Resize(const CResolution2D& Resolution)
	{
		m_Resolution = Resolution;

		m_RunningEstimateXyza.Resize(m_Resolution);
		m_FrameEstimateXyza.Resize(m_Resolution);
		m_FrameBlurXyza.Resize(m_Resolution);
		m_RunningSpecularBloom.Resize(m_Resolution);
		m_EstimateRgbaLdr.Resize(m_Resolution);
		m_DisplayEstimateRgbLdr.Resize(m_Resolution);
		m_RandomSeeds1.Resize(m_Resolution);
		m_RandomSeeds2.Resize(m_Resolution);
		m_NoEstimates.Resize(m_Resolution);
	}

	void Reset(void)
	{
		m_RunningEstimateXyza.Reset();
		m_FrameEstimateXyza.Reset();
		m_FrameBlurXyza.Reset();
		m_RunningSpecularBloom.Reset();
		m_EstimateRgbaLdr.Reset();
		m_DisplayEstimateRgbLdr.Reset();
//		m_RandomSeeds1.Reset();
//		m_RandomSeeds2.Reset();
		m_NoEstimates.Reset();
	}

	void Free(void)
	{
		m_RunningEstimateXyza.Free();
		m_FrameEstimateXyza.Free();
		m_FrameBlurXyza.Free();
		m_RunningSpecularBloom.Free();
		m_EstimateRgbaLdr.Free();
		m_DisplayEstimateRgbLdr.Free();
		m_RandomSeeds1.Free();
		m_RandomSeeds2.Free();
		m_NoEstimates.Free();
	}

	CResolution2D					m_Resolution;
	CCudaBuffer2D<CColorXyza>		m_RunningEstimateXyza;
	CCudaBuffer2D<CColorXyza>		m_FrameEstimateXyza;
	CCudaBuffer2D<CColorXyza>		m_FrameBlurXyza;
	CCudaBuffer2D<CColorXyza>		m_RunningSpecularBloom;
	CCudaBuffer2D<CColorRgbaLdr>	m_EstimateRgbaLdr;
	CCudaBuffer2D<CColorRgbLdr>		m_DisplayEstimateRgbLdr;
	CCudaRandomBuffer2D				m_RandomSeeds1;
	CCudaRandomBuffer2D				m_RandomSeeds2;
	CCudaBuffer2D<int>				m_NoEstimates;
};