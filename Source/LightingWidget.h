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

public slots:
	void OnLoadPreset(const QString& Name);
	void OnSavePreset(const QString& Name);

protected:
	QGridLayout						m_MainLayout;
	QLightSettingsWidget			m_LightSettingsWidget;
	QLightsWidget					m_LightsWidget;
	QBackgroundIlluminationWidget	m_BackgroundIlluminationWidget;
	QTemplateWidget<QLighting>		m_PresetsWidget;
};