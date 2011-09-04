#pragma once

#include <QtGui>

#include "LightsWidget.h"
#include "LightSettingsWidget.h"
#include "BackgroundIlluminationWidget.h"
#include "PresetsWidget.h"
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
	QLightsWidget					m_LightsWidget;
	QLightSettingsWidget			m_LightSettingsWidget;
	QBackgroundIlluminationWidget	m_BackgroundIlluminationWidget;
	QPresetsWidget<QLighting>		m_PresetsWidget;
};