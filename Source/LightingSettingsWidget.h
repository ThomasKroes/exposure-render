#pragma once

#include <QtGui>

#include "Controls.h"

class QLight;

class QLightSettingsWidget : public QGroupBox
{
    Q_OBJECT

public:
    QLightSettingsWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const { return QSize(10, 10); }

public slots:
	void OnLightSelectionChanged(QLight* pLight);
	void OnLockSize(const int& LockHeight);
	void OnThetaChanged(const double& Theta);
	void OnPhiChanged(const double& Phi);
	void OnDistanceChanged(const double& Distance);
	void OnWidthChanged(const double& Width);
	void OnHeightChanged(const double& Height);
	void OnCurrentColorChanged(const QColor& Color);
	void OnIntensityChanged(const double& Intensity);

protected:
	QGridLayout			m_MainLayout;
	QLabel				m_ThetaLabel;
	QDoubleSlider		m_ThetaSlider;
	QDoubleSpinner		m_ThetaSpinBox;
	QLabel				m_PhiLabel;
	QDoubleSlider		m_PhiSlider;
	QDoubleSpinner		m_PhiSpinBox;
	QLabel				m_DistanceLabel;
	QDoubleSlider		m_DistanceSlider;
	QDoubleSpinner		m_DistanceSpinBox;
	QLabel				m_WidthLabel;
	QDoubleSlider		m_WidthSlider;
	QDoubleSpinner		m_WidthSpinBox;
	QLabel				m_HeightLabel;
	QDoubleSlider		m_HeightSlider;
	QDoubleSpinner		m_HeightSpinBox;
	QCheckBox			m_LockSizeCheckBox;
	QLabel				m_ColorLabel;
	QColorPushButton	m_ColorButton;
	QLabel				m_IntensityLabel;
	QDoubleSlider		m_IntensitySlider;
	QDoubleSpinner		m_IntensitySpinBox;
	QLight*				m_pSelectedLight;
};