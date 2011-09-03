#pragma once

#include <QtGui>

class CProjectionWidget : public QGroupBox
{
    Q_OBJECT

public:
    CProjectionWidget(QWidget* pParent = NULL);

private slots:
	void SetFieldOfView(const int& FieldOfView);

private:
	QGridLayout		m_GridLayout;
	QSlider			m_FieldOfViewSlider;
	QSpinBox		m_FieldOfViewSpinBox;
};