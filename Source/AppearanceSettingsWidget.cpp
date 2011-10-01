
// Precompiled headers
#include "Stable.h"

#include "AppearanceSettingsWidget.h"
#include "TransferFunction.h"
#include "RenderThread.h"

QAppearanceSettingsWidget::QAppearanceSettingsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_DensityScaleSlider(),
	m_DensityScaleSpinner(),
	m_ShadingType(),
	m_GradientFactorSlider(),
	m_GradientFactorSpinner(),
	m_IndexOfRefractionSlider(),
	m_IndexOfRefractionSpinner(),
	m_Denoise()
{
	setTitle("Settings");
	
	// Create grid layout
	setLayout(&m_MainLayout);

	// Density scale
	m_MainLayout.addWidget(new QLabel("Density Scale"), 2, 0);

	m_DensityScaleSlider.setOrientation(Qt::Horizontal);
	m_DensityScaleSlider.setRange(0.001, 2.0);
	m_DensityScaleSlider.setValue(1.0);
	m_MainLayout.addWidget(&m_DensityScaleSlider, 2, 1);

	m_DensityScaleSpinner.setRange(0.001, 2.0);
	m_DensityScaleSpinner.setDecimals(3);
	m_MainLayout.addWidget(&m_DensityScaleSpinner, 2, 2);

	m_MainLayout.addWidget(new QLabel("Shading Type"), 3, 0);

	m_ShadingType.addItem("BRDF Only", 0);
	m_ShadingType.addItem("Phase Function Only", 1);
	m_ShadingType.addItem("Hybrid", 2);
	m_MainLayout.addWidget(&m_ShadingType, 3, 1, 1, 2);

	m_MainLayout.addWidget(new QLabel("Gradient Factor"), 4, 0);
	
	m_GradientFactorSlider.setOrientation(Qt::Horizontal);
	m_GradientFactorSlider.setRange(0.001, 100.0);
	m_GradientFactorSlider.setValue(100.0);

	m_MainLayout.addWidget(&m_GradientFactorSlider, 4, 1);

	m_GradientFactorSpinner.setRange(0.001, 100.0);
	m_GradientFactorSpinner.setDecimals(3);

	m_MainLayout.addWidget(&m_GradientFactorSpinner, 4, 2);

	m_MainLayout.addWidget(new QLabel("Index Of Refraction"), 5, 0);

	m_IndexOfRefractionSlider.setOrientation(Qt::Horizontal);
	m_IndexOfRefractionSlider.setRange(0.001, 50.0);
	m_IndexOfRefractionSlider.setValue(10.0);

	m_MainLayout.addWidget(&m_IndexOfRefractionSlider, 5, 1);

	m_IndexOfRefractionSpinner.setRange(0.001, 50.0);
	m_IndexOfRefractionSpinner.setDecimals(3);

	m_MainLayout.addWidget(&m_IndexOfRefractionSpinner, 5, 2);

	m_Denoise.setText("Enable Denoise Filtering");
	m_MainLayout.addWidget(&m_Denoise, 6, 1);

	QObject::connect(&m_DensityScaleSlider, SIGNAL(valueChanged(double)), &m_DensityScaleSpinner, SLOT(setValue(double)));
	QObject::connect(&m_DensityScaleSpinner, SIGNAL(valueChanged(double)), &m_DensityScaleSlider, SLOT(setValue(double)));
	QObject::connect(&m_DensityScaleSlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetDensityScale(double)));

	QObject::connect(&m_GradientFactorSlider, SIGNAL(valueChanged(double)), &m_GradientFactorSpinner, SLOT(setValue(double)));
	QObject::connect(&m_GradientFactorSpinner, SIGNAL(valueChanged(double)), &m_GradientFactorSlider, SLOT(setValue(double)));
	QObject::connect(&m_GradientFactorSlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetMaxGradientMagnitude(double)));

	QObject::connect(&m_IndexOfRefractionSlider, SIGNAL(valueChanged(double)), &m_IndexOfRefractionSpinner, SLOT(setValue(double)));
	QObject::connect(&m_IndexOfRefractionSpinner, SIGNAL(valueChanged(double)), &m_IndexOfRefractionSlider, SLOT(setValue(double)));
	QObject::connect(&m_IndexOfRefractionSlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetIndexOfRefraction(double)));

	QObject::connect(&m_ShadingType, SIGNAL(currentIndexChanged(int)), this, SLOT(OnSetShadingType(int)));
	QObject::connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	QObject::connect(&gStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	QObject::connect(&gTransferFunction, SIGNAL(Changed()), this, SLOT(OnTransferFunctionChanged()));

	QObject::connect(&m_Denoise, SIGNAL(stateChanged(int)), this, SLOT(OnDenoise(int)));
}

void QAppearanceSettingsWidget::OnRenderBegin(void)
{
	m_DensityScaleSlider.setValue(gTransferFunction.GetDensityScale());
	m_ShadingType.setCurrentIndex(0);
	m_Denoise.setChecked(true);

	if (!Scene())
		return;

	m_GradientFactorSlider.setValue(Scene()->m_GradientFactor);
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

void QAppearanceSettingsWidget::OnSetShadingType(int Index)
{
	gTransferFunction.SetShadingType(Index);
}

void QAppearanceSettingsWidget::OnSetMaxGradientMagnitude(double MaxGradMag)
{
	if (!Scene())
		return;

 	Scene()->m_GradientFactor = (float)MaxGradMag;
// 
 	Scene()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void QAppearanceSettingsWidget::OnSetIndexOfRefraction(double IOR)
{
	if (!Scene())
		return;

	Scene()->m_IOR = IOR;
	
	Scene()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void QAppearanceSettingsWidget::OnDenoise(int State)
{
	if (!Scene())
		return;

	Scene()->m_Denoise = (State == 2 ? true : false);

	Scene()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}