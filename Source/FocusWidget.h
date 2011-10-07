#pragma once

#include "Focus.h"

class QFocusWidget : public QGroupBox
{
    Q_OBJECT

public:
    QFocusWidget(QWidget* pParent = NULL);

private slots:
	void SetFocusType(int FocusType);
	void SetFocalDistance(const double& FocalDistance);
	void OnFocusChanged(const QFocus& Focus);

private:
	QGridLayout		m_GridLayout;
	QComboBox		m_FocusTypeComboBox;
	QDoubleSlider	m_FocalDistanceSlider;
	QDoubleSpinner	m_FocalDistanceSpinner;
};