
// Precompiled headers
#include "Stable.h"

#include "Status.h"

// Render status singleton
CStatus gStatus;

void CStatus::SetRenderBegin(void)
{
	emit RenderBegin();
}

void CStatus::SetRenderEnd(void)
{
	emit RenderEnd();
}

void CStatus::SetPreRenderFrame(void)
{
	emit PreRenderFrame();
}

void CStatus::SetPostRenderFrame(void)
{
	emit PostRenderFrame();
}

void CStatus::SetResize(void)
{
	emit Resize();
}

void CStatus::SetLoadPreset(const QString& PresetName)
{
	emit LoadPreset(PresetName);
}

void CStatus::SetStatisticChanged(const QString& Group, const QString& Name, const QString& Value, const QString& Unit /*= ""*/, const QString& Icon /*= ""*/)
{
	emit StatisticChanged(Group, Name, Value, Unit, Icon);
}