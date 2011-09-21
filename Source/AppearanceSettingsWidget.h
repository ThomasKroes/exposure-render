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

private:
	QGridLayout		m_MainLayout;
	QDoubleSlider	m_DensityScaleSlider;
	QDoubleSpinner	m_DensityScaleSpinner;
};