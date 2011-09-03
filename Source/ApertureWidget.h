#pragma once

#include <QtGui>

class CApertureWidget : public QGroupBox
{
    Q_OBJECT

public:
    CApertureWidget(QWidget* pParent = NULL);

private slots:
	void SetAperture(const int& Aperture);

private:
	QGridLayout		m_GridLayout;
	QSlider			m_ApertureSizeSlider;
	QSpinBox		m_ApertureSizeSpinBox;
};