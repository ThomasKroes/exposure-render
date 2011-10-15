#pragma once

#include "Buffer.cuh"

class CCudaModel
{
public:
	CCudaModel(void) :
		m_Intensity(),
		m_GradientMagnitude()
	{
	}

	virtual ~CCudaModel(void)
	{
		Free();
	}

	CCudaModel::CCudaModel(const CCudaModel& Other)
	{
		*this = Other;
	}

	CCudaModel& CCudaModel::operator=(const CCudaModel& Other)
	{
		m_Intensity				= Other.m_Intensity;
		m_GradientMagnitude		= Other.m_GradientMagnitude;

		return *this;
	}

	void Resize(const CResolution3D& Resolution)
	{
		m_Intensity.Resize(Resolution);
		m_GradientMagnitude.Resize(Resolution);
	}

	void Free(void)
	{
		m_Intensity.Free();
		m_GradientMagnitude.Free();
	}

	CCudaBuffer3D<short>	m_Intensity;
	CCudaBuffer3D<short>	m_GradientMagnitude;
};