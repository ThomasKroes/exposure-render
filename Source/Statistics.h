#pragma once

#include "Dll.h"
#include "Defines.h"

#include <string>

class CStatistic
{
public:
	CStatistic(void)
	{
	}

	CStatistic::CStatistic(const CStatistic& Other)
	{
		*this = Other;
	};

	CStatistic& CStatistic::operator=(const CStatistic& Other)
	{
		strcpy_s(m_Name, Other.m_Name);
		strcpy_s(m_Value, Other.m_Value);

		m_Dirty = Other.m_Dirty;

		return *this;
	}

	char	m_Name[255];
	char	m_Value[255];
	bool	m_Dirty;
};

#define MAX_NO_STATISTICS 64

class CStatistics
{
public:
	CStatistic		m_Statistics[MAX_NO_STATISTICS];
	int				m_Count;

	CStatistics(void)
	{
		memset(m_Statistics, 0, MAX_NO_STATISTICS * sizeof(CStatistic));

		m_Count = 0;
	}

	CStatistics::CStatistics(const CStatistics& Other)
	{
		*this = Other;
	};

	CStatistics& CStatistics::operator=(const CStatistics& Other)
	{
		for (int i = 0; i < MAX_NO_STATISTICS; i++)
		{
			m_Statistics[i]	= Other.m_Statistics[i];
		}

		m_Count = Other.m_Count;

		return *this;
	}

	void SetStatistic(const char* pName, const char* pValue)
	{
		if (m_Count >= MAX_NO_STATISTICS)
			return;

		bool Exist = false;

		for (int i = 0; i < m_Count; i++)
		{
			if (strcmp(pName, m_Statistics[i].m_Name) == 0)
			{
				strcpy_s(m_Statistics[i].m_Name, 255, pValue);
				m_Statistics[i].m_Dirty = true;
				Exist = true;
			}
		}

		if (!Exist)
		{
			strcpy_s(m_Statistics[m_Count].m_Name, 255, pName);
			strcpy_s(m_Statistics[m_Count].m_Value, 255, pValue);
			m_Statistics[m_Count].m_Dirty = true;
			m_Count++;
		}
	}

	void ResetDirtyFlags(void)
	{
		for (int i = 0; i < m_Count; i++)
			m_Statistics[i].m_Dirty = false;
	}
};

extern CStatistics gStatistics;