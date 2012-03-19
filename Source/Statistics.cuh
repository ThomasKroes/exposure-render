/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// http://www.johndcook.com/standard_deviation.html

#include "Defines.cuh"

namespace ExposureRender
{

struct RunningStats
{
	DEVICE void Push(float Value, int N)
	{
//		m_n++;

		// See Knuth TAOCP vol 2, 3rd edition, page 232
		if (N == 1)
		{
			m_oldM = m_newM = Value;
			m_oldS = 0.0;
		}
		else
		{
			m_newM = m_oldM + (Value - m_oldM) / (float)N;
			m_newS = m_oldS + (Value - m_oldM) * (Value - m_newM);

			// set up for next iteration
			m_oldM = m_newM; 
			m_oldS = m_newS;
		}
	}

	DEVICE double Mean(int N) const
	{
		return (N > 0) ? m_newM : 0.0f;
	}

	DEVICE double Variance(int N) const
	{
		return ((N > 1) ? m_newS / (float)(N - 1) : 0.0f);
	}

	DEVICE double StandardDeviation(int N) const
	{
		return sqrtf(Variance(N));
	}

	float m_oldM, m_newM, m_oldS, m_newS;
};

}