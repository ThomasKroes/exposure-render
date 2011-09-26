
// Precompiled headers
#include "Stable.h"

#include "Variance.h"

CVariance::CVariance(void) : 
m_Width(0),
	m_Height(0),
	m_pN(NULL),
	m_pOldM(NULL),
	m_pNewM(NULL),
	m_pOldS(NULL),
	m_pNewS(NULL),
	m_pVariance(NULL)
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




