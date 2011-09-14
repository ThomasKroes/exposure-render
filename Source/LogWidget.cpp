
// Precompiled headers
#include "Stable.h"

#include "LogWidget.h"

QTimeTableWidgetItem::QTimeTableWidgetItem(void) :
	QTableWidgetItem(QTime::currentTime().toString("hh:mm:ss"))
{
	setFont(QFont("Arial", 7));
	setTextColor(QColor(60, 60, 60));

	setToolTip(text());
	setStatusTip("Message recorded at " + text());
}

QSize QTimeTableWidgetItem::sizeHint(void) const
{
	return QSize(70, 10);
}

QTableItemMessage::QTableItemMessage(const QString& Message, const QLogger::MessageType& MessageType) :
	QTableWidgetItem(Message)
{
	QString ToolTipPrefix;

	if (MessageType & QLogger::Critical)
		ToolTipPrefix += "Critical error: ";

	setFont(QFont("Arial", 7));
	
	QColor TextColor;

	switch (MessageType)
	{
		case QLogger::Normal:
		{
			TextColor = Qt::blue;
			break;
		}

		case QLogger::Critical:
		{
			TextColor = Qt::red;
			break;
		}
	}
	
	setTextColor(TextColor);

	setToolTip(ToolTipPrefix + Message);
	setStatusTip(ToolTipPrefix + Message);
}

QLogWidget::QLogWidget(QWidget* pParent /*= NULL*/) :
	QTableWidget(pParent),
	m_pLogger(NULL)
{
	setColumnCount(3);
	
	QStringList HeaderLabels;

	HeaderLabels << "time" << "" << "message";

	setHorizontalHeaderLabels(HeaderLabels);
	horizontalHeader()->setResizeMode(0, QHeaderView::ResizeToContents);
	horizontalHeader()->setResizeMode(1, QHeaderView::Fixed);
	horizontalHeader()->setResizeMode(2, QHeaderView::Stretch);
	horizontalHeader()->resizeSection(1, 25);
	horizontalHeader()->setDefaultAlignment(Qt::AlignLeft);
	
	// Disable vertical header
	verticalHeader()->setVisible(false);

	setGridStyle(Qt::NoPen);

	setAlternatingRowColors(true);
}

void QLogWidget::SetLogger(QLogger* pLogger)
{
	m_pLogger = pLogger;

	if (!m_pLogger)
	{
		Log("No valid logger specified!");
		return;
	}
	else
	{
		QObject::connect(m_pLogger, SIGNAL(Log(const QString&, const int&)), this, SLOT(OnLog(const QString&, const int&)));
	}
}

void QLogWidget::OnLog(const QString& Message, const int& Type)
{
	insertRow(0);

	setItem(0, 0, new QTimeTableWidgetItem());
	setItem(0, 1, new QTableWidgetItem(Message));
	setItem(0, 2, new QTableItemMessage(Message, (QLogger::MessageType)Type));
	setRowHeight(0, 18);
}

void QLogWidget::OnClear(void)
{
	if (currentRow() < 0)
		return;

	removeRow(currentRow());
}

void QLogWidget::OnClearAll(void)
{
	clear();
}

void QLogWidget::contextMenuEvent(QContextMenuEvent* pContextMenuEvent)
{
	QMenu ContextMenu(this);
	ContextMenu.setTitle("Log");
	ContextMenu.addAction("Clear", this, SLOT(OnClear()));
	ContextMenu.addAction("Clear All", this, SLOT(OnClearAll()));
	ContextMenu.exec(pContextMenuEvent->globalPos());
}
