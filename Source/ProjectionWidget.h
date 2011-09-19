#pragma once

#include "Projection.h"

class QProjectionWidget : public QGroupBox
{
    Q_OBJECT

public:
    QProjectionWidget(QWidget* pParent = NULL);

private slots:
	void SetFieldOfView(const double& FieldOfView);
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void OnProjectionChanged(const QProjection& Film);

private:
	QGridLayout		m_GridLayout;
	QDoubleSlider	m_FieldOfViewSlider;
	QDoubleSpinner	m_FieldOfViewSpinner;
};