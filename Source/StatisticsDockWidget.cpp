
#include "StatisticsDockWidget.h"
#include "MainWindow.h"

QStatisticsDockWidget::QStatisticsDockWidget(QWidget* pParent) :
	QDockWidget(pParent),
	m_MainLayout(),
	m_StatisticsWidget()
{
	setWindowTitle("Statistics");
	setToolTip("<img src=':/Images/application-list.png'><div>Rendering statistics</div>");
	setWindowIcon(QIcon(":/Images/application-list.png"));

	setWidget(&m_StatisticsWidget);
}

void QStatisticsDockWidget::Init(void)
{
	// Let us know when the rendering begins and ends
// 	connect(gpMainWindow, SIGNAL(RenderBegin()), &m_StatisticsWidget, SLOT(OnRenderBegin()));
// 	connect(gpMainWindow, SIGNAL(RenderEnd()), &m_StatisticsWidget, SLOT(OnRenderEnd()));
}