#pragma once

#include <QtGui>

#include "Controls.h"

class CApertureWidget : public QGroupBox
{
    Q_OBJECT

public:
    CApertureWidget(QWidget* pParent = NULL);

private slots:
	void SetAperture(const double& Aperture);

private:
	QGridLayout		m_GridLayout;
	QDoubleSlider	m_ApertureSizeSlider;
	QDoubleSpinner	m_ApertureSizeSpinBox;
};