
// Precompiled headers
#include "Stable.h"

#include "LogWidget.h"

QLogWidget::QLogWidget(QWidget* pParent /*= NULL*/) :
	QTableWidget(pParent),
	m_pLogger(NULL)
{
	setColumnCount(3);
	
	QStringList HeaderLabels;

	HeaderLabels << "Timestamp" << "" << "Message";

	setHorizontalHeaderLabels(HeaderLabels);
	horizontalHeader()->setResizeMode(0, QHeaderView::Fixed);
	horizontalHeader()->setResizeMode(1, QHeaderView::Fixed);
	horizontalHeader()->setResizeMode(2, QHeaderView::ResizeMode::Stretch);
	horizontalHeader()->resizeSection(0, 100);
	horizontalHeader()->resizeSection(1, 30);
	
	// Disable vertical header
	verticalHeader()->setVisible(false);

	setAlternatingRowColors(true);
}

void QLogWidget::SetLogger(QLogger* pLogger)
{
	m_pLogger = pLogger;

	if (!m_pLogger)
	{
		qDebug("No valid logger specified!");
		return;
	}
	else
	{
		QObject::connect(m_pLogger, SIGNAL(Log(const QString&, const QLogger::MessageType&)), this, SLOT(OnLog(const QString&, const QLogger::MessageType&)));
	}
}

void QLogWidget::OnLog(const QString& Message, const QLogger::MessageType& Type)
{
	insertRow(0);
	setItem(0, 0, new QTableWidgetItem(QTime::currentTime().toString("hh:mm:ss")));
	setItem(0, 1, new QTableWidgetItem(QIcon(""), ""));
	setItem(0, 2, new QTableWidgetItem(Message));
}