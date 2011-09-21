
// Precompiled headers
#include "Stable.h"

#include "AppearanceSettingsWidget.h"
#include "RenderThread.h"

QAppearanceSettingsWidget::QAppearanceSettingsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_DensityScaleSlider(),
	m_DensityScaleSpinner()
{
	setTitle("Settings");
	
	// Create grid layout
	setLayout(&m_MainLayout);

	// Density scale
	m_MainLayout.addWidget(new QLabel("Density Scale"), 2, 0);

	m_DensityScaleSlider.setOrientation(Qt::Horizontal);
	m_DensityScaleSlider.setRange(0.01, 10.0);
	m_DensityScaleSlider.setValue(10.0);
	m_MainLayout.addWidget(&m_DensityScaleSlider, 2, 1);

	m_DensityScaleSpinner.setRange(0.01, 10.0);
	m_MainLayout.addWidget(&m_DensityScaleSpinner, 2, 2);

	connect(&m_DensityScaleSlider, SIGNAL(valueChanged(double)), &m_DensityScaleSpinner, SLOT(setValue(double)));
	connect(&m_DensityScaleSpinner, SIGNAL(valueChanged(double)), &m_DensityScaleSlider, SLOT(setValue(double)));
	connect(&m_DensityScaleSlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetDensityScale(double)));
}

void QAppearanceSettingsWidget::OnSetDensityScale(double DensityScale)
{
	if (!Scene())
		return;

	Scene()->m_DensityScale = DensityScale;

	// Flag the render params as dirty, this will restart the rendering
	Scene()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}