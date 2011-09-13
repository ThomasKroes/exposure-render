
// Precompiled headers
#include "Stable.h"

#include "LogDockWidget.h"

QLogDockWidget::QLogDockWidget(QWidget* pParent /*= NULL*/) :
	QDockWidget(pParent),
	m_LogWidget(this)
{
	setWindowTitle("Log");
	setToolTip("Log");
//	setWindowIcon(QIcon(":/Images/camera.png"));

	m_LogWidget.SetLogger(&gLogger);

	setWidget(&m_LogWidget);
}
