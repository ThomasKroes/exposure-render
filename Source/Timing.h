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

#define MAX_NO_DURATIONS 30

class CTiming
{
public:
	CTiming(void) {};

	HO CTiming(const char* pName)
	{
#ifndef __CUDACC__
		sprintf_s(m_Name, "%s", pName);
#endif
		memset(m_Durations, 0, MAX_NO_DURATIONS * sizeof(float));

		m_NoDurations		= 0;
		m_FilteredDuration	= 0.0f;
	}

		virtual ~CTiming(void) {};

		HO CTiming& CTiming::operator=(const CTiming& Other)
		{
				strcpy_s(m_Name, Other.m_Name);
		
					for (int i = 0; i < MAX_NO_DURATIONS; i++)
					{
							m_Durations[i]	= Other.m_Durations[i];
						}
		
					m_NoDurations		= Other.m_NoDurations;
				m_FilteredDuration	= Other.m_FilteredDuration;
		
					return *this;
			}

	void AddDuration(const float& Duration)
	{
		float TempDurations[MAX_NO_DURATIONS];
		
		memcpy(TempDurations, m_Durations, MAX_NO_DURATIONS * sizeof(float));
		
		m_Durations[0] = Duration;
		
//		m_FilteredDuration = Duration;
//		return;
			
		float SumDuration = Duration;
		
 		for (int i = 0; i < m_NoDurations - 1; i++)
 		{
 			m_Durations[i + 1] = TempDurations[i];
 			SumDuration += TempDurations[i];
 		}
		
		m_FilteredDuration = SumDuration / (float)m_NoDurations;
		
		m_NoDurations = min(MAX_NO_DURATIONS, m_NoDurations + 1);
	}

	char		m_Name[MAX_CHAR_SIZE];
	float		m_Durations[MAX_NO_DURATIONS];
	int			m_NoDurations;
	float		m_FilteredDuration;
};