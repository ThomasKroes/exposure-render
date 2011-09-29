#pragma once

class QAppearanceSettingsWidget : public QGroupBox
{

	Q_OBJECT

public:
    QAppearanceSettingsWidget(QWidget* pParent = NULL);

public slots:
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void OnSetDensityScale(double DensityScale);
	void OnTransferFunctionChanged(void);
	void OnSetShadingType(int Index);
	void OnSetMaxGradientMagnitude(double MaxGradMag);
	void OnSetIndexOfRefraction(double IOR);
	void OnDenoise(int State);

private:
	QGridLayout		m_MainLayout;
	QDoubleSlider	m_DensityScaleSlider;
	QDoubleSpinner	m_DensityScaleSpinner;
	QComboBox		m_ShadingType;
	QDoubleSlider	m_GradientFactorSlider;
	QDoubleSpinner	m_GradientFactorSpinner;
	QDoubleSlider	m_IndexOfRefractionSlider;
	QDoubleSpinner	m_IndexOfRefractionSpinner;
	QCheckBox		m_Denoise;
};