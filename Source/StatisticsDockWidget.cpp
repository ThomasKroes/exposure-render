
// Precompiled headers
#include "Stable.h"

#include "StatisticsDockWidget.h"
#include "MainWindow.h"

QStatisticsDockWidget::QStatisticsDockWidget(QWidget* pParent) :
	QDockWidget(pParent),
	m_MainLayout(),
	m_StatisticsWidget()
{
	setWindowTitle("Statistics");
	setToolTip("<img src=':/Images/application-list.png'><div>Rendering statistics</div>");
	setWindowIcon(GetIcon("application-list"));

	setWidget(&m_StatisticsWidget);
}