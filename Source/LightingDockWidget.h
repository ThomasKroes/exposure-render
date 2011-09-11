#pragma once

#include "LightingWidget.h"

class QLightingDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    QLightingDockWidget(QWidget *parent = NULL);

private:
	QLightingWidget		m_LightingWidget;
};