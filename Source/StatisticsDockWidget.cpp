
// Precompiled headers
#include "Stable.h"

#include "StatisticsDockWidget.h"
#include "MainWindow.h"

class QGraphicsWidget2 : public QGraphicsWidget
{

};

class QTfCanvas : public QWidget
{
public:
	void paintEvent(QPaintEvent * pe)
	{

		QPainter Painter(this);

		if (isEnabled())
			Painter.fillRect(rect(), QBrush(QColor(230, 230, 230)));
		else
			Painter.fillRect(rect(), QBrush(QColor(200, 200, 200)));
	}
};

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