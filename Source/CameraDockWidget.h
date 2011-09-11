#pragma once

#include "CameraWidget.h"

class QCameraDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    QCameraDockWidget(QWidget* pParent = NULL);

private:
	QCameraWidget	m_CameraWidget;
};