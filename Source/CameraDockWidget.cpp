
// Precompiled headers
#include "Stable.h"

#include "CameraDockWidget.h"
#include "MainWindow.h"

QCameraDockWidget::QCameraDockWidget(QWidget* pParent) :
	QDockWidget(pParent),
	m_CameraWidget()
{
	setWindowTitle("Camera");
	setToolTip("<img src=':/Images/camera.png'><div>Camera Properties</div>");
	setWindowIcon(GetIcon("camera"));

	setWidget(&m_CameraWidget);

	QSizePolicy SizePolicy;

	SizePolicy.setVerticalPolicy(QSizePolicy::Maximum);

	setSizePolicy(SizePolicy);
}