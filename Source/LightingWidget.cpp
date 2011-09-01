
#include "LightingWidget.h"

QLightingWidget::QLightingWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_LightingPresetsWidget(),
	m_LightSettingsWidget(),
	m_LightsWidget(),
	m_BackgroundIlluminationWidget()
{
	// Apply main layout
	setLayout(&m_MainLayout);

	// Add widgets
	m_MainLayout.addWidget(&m_LightsWidget, 0, 0);
	m_MainLayout.addWidget(&m_LightSettingsWidget, 1, 0);
	m_MainLayout.addWidget(&m_BackgroundIlluminationWidget, 2, 0);
	m_MainLayout.addWidget(&m_LightingPresetsWidget, 3, 0);

	connect(&m_LightsWidget, SIGNAL(LightSelectionChanged(QLight*)), &m_LightSettingsWidget, SLOT(OnLightSelectionChanged(QLight*)));
}