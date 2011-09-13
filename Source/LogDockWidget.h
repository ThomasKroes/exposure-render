#pragma once

#include "LogWidget.h"

class QLogDockWidget : public QDockWidget
{
	Q_OBJECT

public:
	QLogDockWidget(QWidget* pParent = NULL);

private:
	QLogWidget		m_LogWidget;
};