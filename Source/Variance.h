/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Dll.h"
#include "Defines.h"

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