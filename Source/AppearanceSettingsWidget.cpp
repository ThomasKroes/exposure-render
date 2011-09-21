
// Precompiled headers
#include "Stable.h"

#include "AppearanceSettingsWidget.h"
#include "TransferFunction.h"
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
	m_DensityScaleSlider.setRange(0.001, 1.0);
	m_DensityScaleSlider.setValue(1.0);
	m_MainLayout.addWidget(&m_DensityScaleSlider, 2, 1);

	m_DensityScaleSpinner.setRange(0.001, 1.0);
	m_DensityScaleSpinner.setDecimals(3);
	m_MainLayout.addWidget(&m_DensityScaleSpinner, 2, 2);

	QObject::connect(&m_DensityScaleSlider, SIGNAL(valueChanged(double)), &m_DensityScaleSpinner, SLOT(setValue(double)));
	QObject::connect(&m_DensityScaleSpinner, SIGNAL(valueChanged(double)), &m_DensityScaleSlider, SLOT(setValue(double)));
	QObject::connect(&m_DensityScaleSlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetDensityScale(double)));
	QObject::connect(&gRenderStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	QObject::connect(&gRenderStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	QObject::connect(&gTransferFunction, SIGNAL(Changed()), this, SLOT(OnTransferFunctionChanged()));
}

void QAppearanceSettingsWidget::OnRenderBegin(void)
{
	m_DensityScaleSlider.setValue(gTransferFunction.GetDensityScale());
}

void QAppearanceSettingsWidget::OnRenderEnd(void)
{
	m_DensityScaleSlider.setValue(1.0);
}

void QAppearanceSettingsWidget::OnSetDensityScale(double DensityScale)
{
	gTransferFunction.SetDensityScale(DensityScale);
}

void QAppearanceSettingsWidget::OnTransferFunctionChanged(void)
{
	m_DensityScaleSlider.setValue(gTransferFunction.GetDensityScale(), true);
	m_DensityScaleSpinner.setValue(gTransferFunction.GetDensityScale(), true);
}