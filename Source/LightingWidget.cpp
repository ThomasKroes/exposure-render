
#include "LightingWidget.h"

QLightingWidget::QLightingWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_LightsWidget(),
	m_LightSettingsWidget(),
	m_BackgroundIlluminationWidget(),
	m_PresetsWidget(NULL, "Lighting", "Lighting")
{
	// Apply main layout
	setLayout(&m_MainLayout);

	// Add widgets
	m_MainLayout.addWidget(&m_LightsWidget, 0, 0);
	m_MainLayout.addWidget(&m_LightSettingsWidget, 1, 0);
	m_MainLayout.addWidget(&m_BackgroundIlluminationWidget, 2, 0);
	m_MainLayout.addWidget(&m_PresetsWidget);

	// Connections
	connect(&m_PresetsWidget, SIGNAL(LoadPreset(const QString&)), this, SLOT(OnLoadPreset(const QString&)));
	connect(&m_PresetsWidget, SIGNAL(SavePreset(const QString&)), this, SLOT(OnSavePreset(const QString&)));
	connect(&m_LightsWidget, SIGNAL(LightSelectionChanged(QLight*)), &m_LightSettingsWidget, SLOT(OnLightSelectionChanged(QLight*)));
}

void QLightingWidget::OnLoadPreset(const QString& Name)
{
	gLighting = m_PresetsWidget.GetPreset(Name);
}

void QLightingWidget::OnSavePreset(const QString& Name)
{
	QLighting Preset(gLighting);
	Preset.SetName(Name);

	// Add the preset
	m_PresetsWidget.AddPreset(Preset);
}