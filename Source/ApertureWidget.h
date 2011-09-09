#pragma once

#include <QtGui>

#include "Controls.h"
#include "Aperture.h"

class CApertureWidget : public QGroupBox
{
    Q_OBJECT

public:
    CApertureWidget(QWidget* pParent = NULL);

private slots:
	void SetAperture(const double& Aperture);
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void OnApertureChanged(const QAperture& Aperture);

private:
	QGridLayout		m_GridLayout;
	QDoubleSlider	m_SizeSlider;
	QDoubleSpinner	m_SizeSpinBox;
};