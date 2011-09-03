
#include "CameraDockWidget.h"
#include "MainWindow.h"
#include "Scene.h"

QCameraDockWidget::QCameraDockWidget(QWidget* pParent) :
	QDockWidget(pParent),
	m_CameraWidget()
{
	setWindowTitle("Camera");
	setToolTip("<img src=':/Images/camera.png'><div>Camera Properties</div>");
	setWindowIcon(QIcon(":/Images/camera.png"));

	setWidget(&m_CameraWidget);
}