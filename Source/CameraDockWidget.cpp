
#include "CameraDockWidget.h"
#include "MainWindow.h"
#include "Scene.h"

QCameraDockWidget::QCameraDockWidget(QWidget* pParent) :
	QDockWidget(pParent),
	m_CameraWidget()
{
	setWindowTitle("Camera");
	setToolTip("Camera settings");

	setWidget(&m_CameraWidget);
}