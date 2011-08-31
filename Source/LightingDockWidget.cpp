
#include "LightingDockWidget.h"

QLightingDockWidget::QLightingDockWidget(QWidget* pParent) :
	QDockWidget(pParent),
	m_LightingWidget()
{
	// Window title and tooltip
	setWindowTitle("Lighting");
	setToolTip("Lighting configuration");

	// Apply widget
	setWidget(&m_LightingWidget);
}