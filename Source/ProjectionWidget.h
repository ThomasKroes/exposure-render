#pragma once

#include <QtGui>

#include "Controls.h"
#include "Projection.h"

class CProjectionWidget : public QGroupBox
{
    Q_OBJECT

public:
    CProjectionWidget(QWidget* pParent = NULL);

private slots:
	void SetFieldOfView(const double& FieldOfView);
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void OnProjectionChanged(const QProjection& Film);

private:
	QGridLayout		m_GridLayout;
	QDoubleSlider	m_FieldOfViewSlider;
	QDoubleSpinner	m_FieldOfViewSpinBox;
};