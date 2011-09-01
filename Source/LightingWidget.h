#pragma once

#include <QtGui>

#include "LightingPresetsWidget.h"
#include "LightingSettingsWidget.h"
#include "LightsWidget.h"
#include "BackgroundIlluminationWidget.h"

class QLightingWidget : public QWidget
{
    Q_OBJECT

public:
    QLightingWidget(QWidget* pParent = NULL);

protected:
	QGridLayout						m_MainLayout;
	QLightingPresetsWidget			m_LightingPresetsWidget;
	QLightSettingsWidget			m_LightSettingsWidget;
	QLightsWidget					m_LightsWidget;
	QBackgroundIlluminationWidget	m_BackgroundIlluminationWidget;
};