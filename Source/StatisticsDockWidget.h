#pragma once

#include <QtGui>

#include "StatisticsWidget.h"

class QStatisticsDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    QStatisticsDockWidget(QWidget* pParent = 0);

	void Init(void);

private:
	QGridLayout			m_MainLayout;
	QStatisticsWidget	m_StatisticsWidget;
};