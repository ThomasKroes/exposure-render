#pragma once

#include <QtGui>
#include <QVTKWidget.h>

class CVtkWidget : public QWidget
{
    Q_OBJECT

public:
    CVtkWidget(QWidget* pParent = NULL);
	
	QVTKWidget*		GetQtVtkWidget(void);

private:
	QGridLayout		m_MainLayout;
	QVTKWidget		m_QtVtkWidget;
};