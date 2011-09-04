#pragma once

#include <QtGui>

#include "Controls.h"

class CFocusWidget : public QGroupBox
{
    Q_OBJECT

public:
    CFocusWidget(QWidget* pParent = NULL);

private slots:
	void SetFocusType(const int& FocusType);
	void SetFocalDistance(const int& FocalDistance);

private:
	QGridLayout		m_GridLayout;
	QComboBox		m_FocusTypeComboBox;
	QDoubleSlider	m_FocalDistanceSlider;
	QDoubleSpinner	m_FocalDistanceSpinBox;
};