#pragma once

#include <QtGui>
#include <QVTKWidget.h>

class CVtkWidget : public QWidget
{
    Q_OBJECT

public:
    CVtkWidget(QWidget* pParent = NULL);
	QVTKWidget* GetQtVtkWidget(void) { return m_pQtVtkWidget; };

private:
	QVTKWidget*		m_pQtVtkWidget;
	QVBoxLayout*	m_pMainLayout;
	QHBoxLayout*	m_pTopLayout;
	QHBoxLayout*	m_pBottomLayout;

	QLabel*			m_pZoomLabel;
	QSlider*		m_pZoomSlider;
	QComboBox*		m_pZoomComboBox;
};