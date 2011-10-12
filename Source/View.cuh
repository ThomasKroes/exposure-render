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

	~CCudaBuffer2D(void)
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

		HandleCudaError(cudaMallocPitch(&m_pData, &m_Pitch, m_Resolution.GetResX(), m_Resolution.GetResY()), "Reset Cuda Buffer");
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

	CResolution2D	m_Resolution;
	T*				m_pData;
	size_t			m_Pitch;
};

class CCudaView
{
public:
	CCudaView(void) :
		m_Resolution(0, 0),
		m_EstimateXyza(),
		m_EstimateFrameXyza(),
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
		m_EstimateXyza				= Other.m_EstimateXyza;
		m_EstimateFrameXyza			= Other.m_EstimateFrameXyza;
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

		m_EstimateXyza.Resize(m_Resolution);
		m_EstimateFrameXyza.Resize(m_Resolution);
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
		m_EstimateXyza.Reset();
		m_EstimateFrameXyza.Reset();
		m_FrameBlurXyza.Reset();
		m_RunningSpecularBloom.Reset();
		m_EstimateRgbaLdr.Reset();
		m_DisplayEstimateRgbLdr.Reset();
		m_RandomSeeds1.Reset();
		m_RandomSeeds2.Reset();
		m_NoEstimates.Reset();
	}

	void Free(void)
	{
		m_EstimateXyza.Free();
		m_EstimateFrameXyza.Free();
		m_FrameBlurXyza.Free();
		m_RunningSpecularBloom.Free();
		m_EstimateRgbaLdr.Free();
		m_DisplayEstimateRgbLdr.Free();
		m_RandomSeeds1.Free();
		m_RandomSeeds2.Free();
		m_NoEstimates.Free();
	}

	void BindTextures(void)
	{
		/*
		size_t Offset;

		cudaChannelFormatDesc ChannelDesc;
		
		ChannelDesc = cudaCreateChannelDesc<float4>();

		HandleCudaError(cudaBindTexture2D(&Offset, gTexRunningEstimateXyza, (void*)m_EstimateXyza.m_pData, &ChannelDesc, m_Resolution.GetResX(), m_Resolution.GetResY(), m_EstimateXyza.m_Pitch), "Bind Estimate XYZA");
		HandleCudaError(cudaBindTexture2D(&Offset, gTexFrameEstimateXyza, (void*)m_EstimateFrameXyza.m_pData, &ChannelDesc, m_Resolution.GetResX(), m_Resolution.GetResY(), m_EstimateFrameXyza.m_Pitch), "Bind Frame Estimate XYZA");
		HandleCudaError(cudaBindTexture2D(&Offset, gTexFrameBlurXyza, (void*)m_FrameBlurXyza.m_pData, &ChannelDesc, m_Resolution.GetResX(), m_Resolution.GetResY(), m_FrameBlurXyza.m_Pitch), "Bind Frame Blur XYZA");
		HandleCudaError(cudaBindTexture2D(&Offset, gTexRunningSpecularBloomXyza, (void*)m_RunningSpecularBloom.m_pData, &ChannelDesc, m_Resolution.GetResX(), m_Resolution.GetResY(), m_RunningSpecularBloom.m_Pitch), "Bind Running Specular Bloom");

		ChannelDesc = cudaCreateChannelDesc<uchar4>();

		HandleCudaError(cudaBindTexture2D(&Offset, gTexRunningEstimateRgba, (void*)m_EstimateRgbaLdr.m_pData, &ChannelDesc, m_Resolution.GetResX(), m_Resolution.GetResY(), m_EstimateRgbaLdr.m_Pitch), "Bind Estimate RGBA");
		*/
	}

	CResolution2D					m_Resolution;
	CCudaBuffer2D<CColorXyza>		m_EstimateXyza;
	CCudaBuffer2D<CColorXyza>		m_EstimateFrameXyza;
	CCudaBuffer2D<CColorXyza>		m_FrameBlurXyza;
	CCudaBuffer2D<CColorXyza>		m_RunningSpecularBloom;
	CCudaBuffer2D<CColorRgbaLdr>	m_EstimateRgbaLdr;
	CCudaBuffer2D<CColorRgbLdr>		m_DisplayEstimateRgbLdr;
	CCudaBuffer2D<int>				m_RandomSeeds1;
	CCudaBuffer2D<int>				m_RandomSeeds2;
	CCudaBuffer2D<int>				m_NoEstimates;
};