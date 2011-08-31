
#include "LightingWidget.h"

QLightingWidget::QLightingWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout()
{
	// Apply main layout
	setLayout(&m_MainLayout);

	// Add widgets
	m_MainLayout.addWidget(&m_LightingPresetsWidget, 0, 0);
	m_MainLayout.addWidget(&m_LightsWidget, 0, 1);
	m_MainLayout.addWidget(&m_LightSettingsWidget, 1, 0, 1, 2);

	connect(&m_LightsWidget, SIGNAL(LightSelectionChanged(QLight*)), &m_LightSettingsWidget, SLOT(OnLightSelectionChanged(QLight*)));
}