#pragma once

#include <QtGui>

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
	QSlider			m_FocalDistanceSlider;
	QSpinBox		m_FocalDistanceSpinBox;
};