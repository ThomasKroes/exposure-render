
#include "LightingPresetsWidget.h"
#include "RenderThread.h"

QLightingPresetsWidget::QLightingPresetsWidget(QWidget* pParent) :
	QPresetsWidget("LightingPresets", pParent)
{
	// Title, status and tooltip
	setTitle("Presets");
	setToolTip("Lighting Presets");
	setStatusTip("Lighting Presets");
}

void QLightingPresetsWidget::LoadPresetsFromFile(const bool& ChoosePath /*= false*/)
{

}

void QLightingPresetsWidget::SavePresetsToFile(const bool& ChoosePath /*= false*/)
{

}
