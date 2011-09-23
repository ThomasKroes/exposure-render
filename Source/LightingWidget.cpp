
// Precompiled headers
#include "Stable.h"

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
	m_MainLayout.addWidget(&m_LightsWidget, 1, 0);
	m_MainLayout.addWidget(&m_LightSettingsWidget, 2, 0);
	m_MainLayout.addWidget(&m_BackgroundIlluminationWidget, 3, 0);
	m_MainLayout.addWidget(&m_PresetsWidget, 0, 0);

	QObject::connect(&m_PresetsWidget, SIGNAL(LoadPreset(const QString&)), this, SLOT(OnLoadPreset(const QString&)));
	QObject::connect(&gStatus, SIGNAL(LoadPreset(const QString&)), &m_PresetsWidget, SLOT(OnLoadPreset(const QString&)));
	QObject::connect(&m_PresetsWidget, SIGNAL(SavePreset(const QString&)), this, SLOT(OnSavePreset(const QString&)));
}

void QLightingWidget::OnLoadPreset(const QString& Name)
{
	// Only load the preset when it exists
	if (!m_PresetsWidget.HasPreset(Name))
		return;

	gLighting = m_PresetsWidget.GetPreset(Name);
}

void QLightingWidget::OnSavePreset(const QString& Name)
{
	QLighting Preset(gLighting);
	Preset.SetName(Name);

	// Add the preset
	m_PresetsWidget.SavePreset(Preset);
}