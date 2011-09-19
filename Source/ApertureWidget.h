#pragma once

#include "Aperture.h"

class QApertureWidget : public QGroupBox
{
    Q_OBJECT

public:
    QApertureWidget(QWidget* pParent = NULL);

public slots:
	void SetAperture(const double& Aperture);
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void OnApertureChanged(const QAperture& Aperture);

private:
	QGridLayout		m_GridLayout;
	QDoubleSlider	m_SizeSlider;
	QDoubleSpinner	m_SizeSpinner;
};