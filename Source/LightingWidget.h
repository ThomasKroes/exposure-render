#pragma once

#include "LightsWidget.h"
#include "LightWidget.h"
#include "BackgroundWidget.h"
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
	QGridLayout					m_MainLayout;
	QLightsWidget				m_LightsWidget;
	QLightWidget				m_LightSettingsWidget;
	QBackgroundWidget			m_BackgroundIlluminationWidget;
	QPresetsWidget<QLighting>	m_PresetsWidget;
};