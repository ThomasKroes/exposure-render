
#include "LightingPresetsWidget.h"
#include "RenderThread.h"

QLightingPresetsWidget::QLightingPresetsWidget(QWidget* pParent) :
	QPresetsWidget(pParent)
{
	// Title, status and tooltip
	setTitle("Presets");
	setToolTip("Lighting Presets");
	setStatusTip("Lighting Presets");
}