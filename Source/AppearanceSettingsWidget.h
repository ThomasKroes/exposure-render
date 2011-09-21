#pragma once

class QAppearanceSettingsWidget : public QGroupBox
{

	Q_OBJECT

public:
    QAppearanceSettingsWidget(QWidget* pParent = NULL);

private slots:
	void OnSetDensityScale(double DensityScale);

private:
	QGridLayout		m_MainLayout;
	QDoubleSlider	m_DensityScaleSlider;
	QDoubleSpinner	m_DensityScaleSpinner;
};