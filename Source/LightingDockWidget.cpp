
// Precompiled headers
#include "Stable.h"

#include "LightingDockWidget.h"

QLightingDockWidget::QLightingDockWidget(QWidget* pParent) :
	QDockWidget(pParent),
	m_LightingWidget()
{
	// Window title and tooltip
	setWindowTitle("Lighting");
	setToolTip("<img src=':/Images/light-bulb.png'><div>Lighting Properties</div>");
	setWindowIcon(GetIcon("light-bulb"));

	// Apply widget
	setWidget(&m_LightingWidget);
}