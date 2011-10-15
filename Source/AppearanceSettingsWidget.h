#pragma once

class QAppearanceSettingsWidget : public QGroupBox
{
	Q_OBJECT

public:
    QAppearanceSettingsWidget(QWidget* pParent = NULL);

public slots:
	void OnRenderBegin(void);
	void OnSetDensityScale(double DensityScale);
	void OnTransferFunctionChanged(void);
	void OnSetShadingType(int Index);
	void OnSetGradientFactor(double GradientFactor);
	void OnSetStepSizePrimaryRay(const double& StepSizePrimaryRay);
	void OnSetStepSizeSecondaryRay(const double& StepSizeSecondaryRay);

private:
	QGridLayout		m_MainLayout;
	QDoubleSlider	m_DensityScaleSlider;
	QDoubleSpinner	m_DensityScaleSpinner;
	QComboBox		m_ShadingType;
	QDoubleSlider	m_GradientFactorSlider;
	QDoubleSpinner	m_GradientFactorSpinner;

	QDoubleSlider	m_StepSizePrimaryRaySlider;
	QDoubleSpinner	m_StepSizePrimaryRaySpinner;

	QDoubleSlider	m_StepSizeSecondaryRaySlider;
	QDoubleSpinner	m_StepSizeSecondaryRaySpinner;
};