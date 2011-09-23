
// Precompiled headers
#include "Stable.h"

#include "Status.h"

// Render status singleton
CRenderStatus gRenderStatus;

void CRenderStatus::StatisticChanged(const QString& Group, const QString& Name, const QString& Value, const QString& Unit /*= ""*/, const QString& Icon /*= ""*/)
{
	emit StatisticChanged(Group, Name, Value, Unit, Icon);
}
