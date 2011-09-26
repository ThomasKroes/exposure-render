#pragma once

#include "Dll.h"
#include "Defines.h"

class EXPOSURE_RENDER_DLL CVariance
{
public:
	CVariance(void);
	virtual ~CVariance(void);
		
	void Free(void);
	void Resize(int Width, int Height);
	void Reset(void);

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

		m_pVariance[ID] = Variance(ID);
	}

	HOD int NumDataValues(int ID) const
	{
		return m_pN[ID];
	}

	HOD float Mean(int ID) const
	{
		return (m_pN[ID] > 0) ? m_pNewM[ID] : 0.0;
	}

	HOD float Variance(int ID) const
	{
		return ( (m_pN[ID] > 1) ? m_pNewS[ID]/(m_pN[ID] - 1) : 0.0f );
	}

	HOD float StandardDeviation(int ID) const
	{
		return sqrt( Variance(ID) );
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
};