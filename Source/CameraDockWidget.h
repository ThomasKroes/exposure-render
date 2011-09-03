#pragma once

#include <QtGui>

#include "CameraWidget.h"

class QCameraDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    QCameraDockWidget(QWidget* pParent = NULL);

private:
	CCameraWidget	m_CameraWidget;
};