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