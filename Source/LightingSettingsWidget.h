#pragma once

#include <QtGui>

#include "ColorButtonWidget.h"

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
	void OnThetaChanged(const int& Theta);
	void OnPhiChanged(const int& Phi);
	void OnDistanceChanged(const int& Distance);
	void OnWidthChanged(const int& Width);
	void OnHeightChanged(const int& Height);
	void OnCurrentColorChanged(const QColor& Color);
	void OnIntensityChanged(const int& Intensity);

protected:
	QGridLayout			m_MainLayout;
	QLabel				m_ThetaLabel;
	QSlider				m_ThetaSlider;
	QSpinBox			m_ThetaSpinBox;
	QLabel				m_PhiLabel;
	QSlider				m_PhiSlider;
	QSpinBox			m_PhiSpinBox;
	QLabel				m_DistanceLabel;
	QSlider				m_DistanceSlider;
	QSpinBox			m_DistanceSpinBox;
	QLabel				m_WidthLabel;
	QSlider				m_WidthSlider;
	QSpinBox			m_WidthSpinBox;
	QLabel				m_HeightLabel;
	QSlider				m_HeightSlider;
	QSpinBox			m_HeightSpinBox;
	QCheckBox			m_LockSizeCheckBox;
	QLabel				m_ColorLabel;
	QColorPushButton	m_ColorButton;
	QGridLayout			m_ColorLayout;
	QLabel				m_IntensityLabel;
	QSlider				m_IntensitySlider;
	QSpinBox			m_IntensitySpinBox;
	QLight*				m_pSelectedLight;
};