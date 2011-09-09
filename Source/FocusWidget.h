#pragma once

#include <QtGui>

#include "Controls.h"
#include "Focus.h"

class CFocusWidget : public QGroupBox
{
    Q_OBJECT

public:
    CFocusWidget(QWidget* pParent = NULL);

private slots:
	void SetFocalDistance(const double& FocalDistance);
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void OnFocusChanged(const QFocus& Focus);

private:
	QGridLayout		m_GridLayout;
	QComboBox		m_FocusTypeComboBox;
	QDoubleSlider	m_FocalDistanceSlider;
	QDoubleSpinner	m_FocalDistanceSpinBox;
};