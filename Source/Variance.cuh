/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Geometry.h"
#include "Variance.h"
#include "CudaUtilities.h"

/*
class EXPOSURE_RENDER_DLL CVariance
{
public:
	CVariance(void);
	virtual ~CVariance(void);
		
	void	Free(void);
	void	Resize(int Width, int Height);
	void	Reset(void);
	float*	GetVarianceBuffer(void);
	float	GetMeanVariance(void) const;
	void	SetMeanVariance(const float& Variance);

	HOD void Push(float x, int ID)
	{
		m_pN[ID]++;

		// See Knuth TAOCP vol 2, 3rd edition, page 232
		if (m_pN[ID] == 1)
		{
			m_pOldM[ID] = m_pNewM[ID] = x;
			m_pOldS[ID] = 0.0;
		}
		else
		{
			m_pNewM[ID] = m_pOldM[ID] + (x - m_pOldM[ID])/m_pN[ID];
			m_pNewS[ID] = m_pOldS[ID] + (x - m_pOldM[ID])*(x - m_pNewM[ID]);

			// set up for next iteration
			m_pOldM[ID] = m_pNewM[ID]; 
			m_pOldS[ID] = m_pNewS[ID];
		}

		m_pVariance[ID] = GetVariance(ID);
	}

	HOD int NumDataValues(int ID) const
	{
		return m_pN[ID];
	}

	HOD float Mean(int ID) const
	{
		return (m_pN[ID] > 0) ? m_pNewM[ID] : 0.0;
	}

	HOD float GetVariance(int ID) const
	{
		return ( (m_pN[ID] > 1) ? m_pNewS[ID]/(m_pN[ID] - 1) : 0.0f );
	}

	HOD float StandardDeviation(int ID) const
	{
		return sqrt( GetVariance(ID) );
	}

private:
	int			m_Width;
	int			m_Height;
	int*		m_pN;
	float*		m_pOldM;
	float*		m_pNewM;
	float*		m_pOldS;
	float*		m_pNewS;
	float*		m_pVariance;
	float		m_MeanVariance;
};

CVariance::CVariance(void) : 
	m_Width(0),
	m_Height(0),
	m_pN(NULL),
	m_pOldM(NULL),
	m_pNewM(NULL),
	m_pOldS(NULL),
	m_pNewS(NULL),
	m_pVariance(NULL),
	m_MeanVariance(0)
{
}

CVariance::~CVariance(void)
{
	Free();
}

void CVariance::Free(void)
{
	cudaFree(m_pN);
	cudaFree(m_pOldM);
	cudaFree(m_pNewM);
	cudaFree(m_pOldS);
	cudaFree(m_pNewS);
	cudaFree(m_pVariance);
}

void CVariance::Resize(int Width, int Height)
{
	if (Width == 0 || Height == 0)
		return;

	m_Width		= Width;
	m_Height	= Height;

	const int NoElements = m_Width * m_Height;

	m_MeanVariance = 0;

	Free();

	gStatus.SetStatisticChanged("CUDA Memory", "Variance", "", "MB");

	cudaMalloc((void**)&m_pN, NoElements * sizeof(int));
	gStatus.SetStatisticChanged("Variance", "N buffer", QString::number((float)NoElements * sizeof(int) / MB, 'f', 2), "MB");

	cudaMalloc((void**)&m_pOldM, NoElements * sizeof(float));
	gStatus.SetStatisticChanged("Variance", "Old Mean Buffer", QString::number((float)NoElements * sizeof(float) / MB, 'f', 2), "MB");

	cudaMalloc((void**)&m_pNewM, NoElements * sizeof(float));
	gStatus.SetStatisticChanged("Variance", "New Mean Buffer", QString::number((float)NoElements * sizeof(float) / MB, 'f', 2), "MB");

	cudaMalloc((void**)&m_pOldS, NoElements * sizeof(float));
	gStatus.SetStatisticChanged("Variance", "Old S Buffer", QString::number((float)NoElements * sizeof(float) / MB, 'f', 2), "MB");

	cudaMalloc((void**)&m_pNewS, NoElements * sizeof(float));
	gStatus.SetStatisticChanged("Variance", "New S Buffer", QString::number((float)NoElements * sizeof(float) / MB, 'f', 2), "MB");

	cudaMalloc((void**)&m_pVariance, NoElements * sizeof(float));
	gStatus.SetStatisticChanged("Variance", "Magnitude Buffer", QString::number((float)NoElements * sizeof(float) / MB, 'f', 2), "MB");

	Reset();
}

void CVariance::Reset(void)
{
	const int NoElements = m_Width * m_Height;

	cudaMemset(m_pN, 0, NoElements * sizeof(int));
	cudaMemset(m_pOldM, 0, NoElements * sizeof(int));
	cudaMemset(m_pNewM, 0, NoElements * sizeof(float));
	cudaMemset(m_pOldS, 0, NoElements * sizeof(float));
	cudaMemset(m_pNewS, 0, NoElements * sizeof(float));
	cudaMemset(m_pVariance, 0, NoElements * sizeof(float));
}

float* CVariance::GetVarianceBuffer(void)
{
	return m_pVariance;
}

float CVariance::GetMeanVariance(void) const
{
	return m_MeanVariance;
}

void CVariance::SetMeanVariance(const float& Variance)
{
	m_MeanVariance = Variance;
}
*/

KERNEL void KrnlComputeVariance(int Width, int Height, CColorXyz* gpEstXyz, CColorXyz* pAccEstXyz, float N, float Exposure, CColorRgbaLdr* pPixels)
{
}

void ComputeVariance(int Width, int Height, CColorXyz* pEstFrameXyz, CColorXyz* pAccEstXyz, float N, float Exposure, CColorRgbaLdr* pPixels)
{
	const dim3 KernelBlock(16, 8);
	const dim3 KernelGrid((int)ceilf((float)Width / (float)KernelBlock.x), (int)ceilf((float)Height / (float)KernelBlock.y));

	KrnlComputeVariance<<<KernelGrid, KernelBlock>>>(Width, Height, pEstFrameXyz, pAccEstXyz, N, Exposure, pPixels);
	HandleCudaError(cudaGetLastError());
}