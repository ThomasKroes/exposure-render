#pragma once

#include <QtGui>

#include "LightsWidget.h"
#include "LightingSettingsWidget.h"
#include "PresetsWidget.h"
#include "BackgroundIlluminationWidget.h"
#include "Lighting.h"

class QLightingWidget : public QWidget
{
    Q_OBJECT

public:
    QLightingWidget(QWidget* pParent = NULL);

protected:
	QGridLayout						m_MainLayout;
	QLightSettingsWidget			m_LightSettingsWidget;
	QLightsWidget					m_LightsWidget;
	QBackgroundIlluminationWidget	m_BackgroundIlluminationWidget;
	QPresetsWidget					m_LightingPresetsWidget;
	QPresets<QLighting>				m_Presets;
};