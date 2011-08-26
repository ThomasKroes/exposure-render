#pragma once

#include "Dll.h"
#include "Defines.h"

#include <map>

using namespace std;

#define MAX_NO_DURATIONS 100


class FIDELITY_RENDER_DLL CEvent
{
public:
	CEvent(void) {};

	HO CEvent(const char* pName)
	{
#ifndef __CUDACC__
		sprintf_s(m_Name, "%s", pName);
#endif
		memset(m_Durations, 0, MAX_NO_DURATIONS * sizeof(float));

		m_NoDurations		= 0;
		m_FilteredDuration	= 0.0f;
	}

	virtual ~CEvent(void) {};

	CEvent& CEvent::operator=(const CEvent& Other)
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

		m_NoDurations++;

		m_NoDurations = min(MAX_NO_DURATIONS, m_NoDurations);
	}

	char		m_Name[MAX_CHAR_SIZE];
	float		m_Durations[MAX_NO_DURATIONS];
	int			m_NoDurations;
	float		m_FilteredDuration;
};
/**/

class FIDELITY_RENDER_DLL CStatistics
{
public:
	double			m_Offset;

	static double	m_LineSpacing;
	static double	m_TabName;
	static double	m_TabColon;
	static double	m_TabValue;
	static double	m_TabUnit;

	CStatistics(void) :
		m_Offset(0.0)
	{
	}

	void Add(char* pName, char* pValue = NULL, char* pUnit = NULL, int Color = 0, int Bold = 0, int Italic = 0)
	{
		/*
		vtkTextProperty* pTextProperty = NULL;

		double TextColor[3];

		switch (Color)
		{
			case 0:
			{
				TextColor[0]	= 0.8;
				TextColor[1]	= 0.6;
				TextColor[2]	= 0.1;
				break;
			}

			case 1:
			{
				TextColor[0]	= 0.8;
				TextColor[1]	= 0.1;
				TextColor[2]	= 0.1;
				break;
			}

			case 2:
			{
				TextColor[0]	= 0.1;
				TextColor[1]	= 0.9;
				TextColor[2]	= 0.2;
				break;
			}
		}

		m_Offset += m_LineSpacing;

		// Name
		m_Name[pName] = vtkTextActor::New();

		// Get text property
		pTextProperty = m_Name[pName]->GetTextProperty();

		// Configure actor
		m_Name[pName]->SetInput(pName);
		m_Name[pName]->SetDisplayPosition(m_TabName, m_Offset);

		// Set properties
		pTextProperty->SetFontFamilyToArial();
		pTextProperty->SetFontSize(10);
		pTextProperty->SetColor(TextColor);
// 		pTextProperty->SetItalic(Italic);
// 		pTextProperty->SetBold(Bold);

		// Add actor
		m_pRenderer->AddActor2D(m_Name[pName]);

		// Value
		if (pValue)
		{
			// Colon
			m_Colon[pName] = vtkTextActor::New();

			// Get text property
			pTextProperty = m_Colon[pName]->GetTextProperty();

			// Configure actor
			m_Colon[pName]->SetInput(":");
			m_Colon[pName]->SetDisplayPosition(m_TabColon, m_Offset);

			// Set properties
			pTextProperty->SetFontFamilyToArial();
			pTextProperty->SetFontSize(10);
			pTextProperty->SetColor(TextColor);
 			pTextProperty->SetItalic(Italic);
// 			pTextProperty->SetBold(Bold);

			// Add actor
			m_pRenderer->AddActor2D(m_Colon[pName]);

			m_Value[pName] = vtkTextActor::New();

			// Get text property
			pTextProperty = m_Value[pName]->GetTextProperty();

			// Configure actor
			m_Value[pName]->SetInput(pValue);
			m_Value[pName]->SetDisplayPosition(m_TabValue, m_Offset);

			// Set properties
			pTextProperty->SetFontFamilyToArial();
			pTextProperty->SetFontSize(10);
			pTextProperty->SetColor(TextColor);
// 			pTextProperty->SetItalic(Italic);
// 			pTextProperty->SetBold(Bold);
		}

		// Add actor
		m_pRenderer->AddActor2D(m_Value[pName]);

		// Unit
		if (pUnit)
		{
			m_Unit[pName] = vtkTextActor::New();

			// Get text property
			pTextProperty = m_Unit[pName]->GetTextProperty();

			// Configure actor
			m_Unit[pName]->SetInput(pUnit);
			m_Unit[pName]->SetDisplayPosition(m_TabUnit, m_Offset);

			// Set properties
			pTextProperty->SetFontFamilyToArial();
			pTextProperty->SetFontSize(10);
			pTextProperty->SetColor(TextColor);
// 			pTextProperty->SetItalic(Italic);
// 			pTextProperty->SetBold(Bold);

			// Add actor
			m_pRenderer->AddActor2D(m_Unit[pName]);
		}
		*/
	}

	void SetName(const char* pID, const char* pValue)
	{
		/*
 		if (m_Name.find(pID) != m_Name.end())
 			m_Name[pID]->SetInput(pValue);
		*/
	}

	void SetValue(const char* pID, const char* pValue)
	{
		/*
		if (m_Value.find(pID) != m_Value.end())
			m_Value[pID]->SetInput(pValue);
		*/
	}

	void SetUnit(const char* pID, const char* pValue)
	{
		/*
		if (m_Unit.find(pID) != m_Unit.end())
			m_Unit[pID]->SetInput(pValue);
		*/
	}

	void SetBoldValue(const char* pID, int Bold)
	{
		/*
		if (m_Value.find(pID) != m_Value.end())
		{
			vtkTextProperty* pTextProperty = m_Value[pID]->GetTextProperty();

			pTextProperty->SetBold(Bold);
		}
		*/
	}

	void SetItalicValue(const char* pID, int Italic)
	{
		/*
		if (m_Value.find(pID) != m_Value.end())
		{
			vtkTextProperty* pTextProperty = m_Value[pID]->GetTextProperty();

			pTextProperty->SetItalic(Italic);
		}
		*/
	}
};