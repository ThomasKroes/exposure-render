
// Precompiled headers
#include "Stable.h"

#include "LogWidget.h"
#include "RenderThread.h"

QTimeTableWidgetItem::QTimeTableWidgetItem(void) :
	QTableWidgetItem(QTime::currentTime().toString("hh:mm:ss"))
{
	setFont(QFont("Arial", 8));
//	setTextColor(QColor(60, 60, 60));

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

	setFont(QFont("Arial", 8));
	
	QColor TextColor;

	switch (MessageType)
	{
		case QLogger::Normal:
		{
			TextColor = Qt::black;
			break;
		}

		case QLogger::Critical:
		{
			TextColor = Qt::red;
			break;
		}
	}
	
//	setTextColor(TextColor);

	setToolTip(ToolTipPrefix + Message);
	setStatusTip(ToolTipPrefix + Message);
}

QTableItemProgress::QTableItemProgress(const QString& Event, const float& Progress)
{
	QString ProgressString = Event;
	
	if (Progress == 100.0f)
		ProgressString += "... Done";
	else
		ProgressString += QString::number(Progress, 'f', 2);

	setText(ProgressString);
	setFont(QFont("Arial", 7));
//	setTextColor(Qt::blue);
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
	horizontalHeader()->setVisible(false);

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
		QObject::connect(m_pLogger, SIGNAL(Log(const QString&, const QString&)), this, SLOT(OnLog(const QString&, const QString&)));
		QObject::connect(m_pLogger, SIGNAL(LogProgress(const QString&, const float&)), this, SLOT(OnLogProgress(const QString&, const float&)));
	}
}

void QLogWidget::OnLog(const QString& Message, const int& Type)
{
	insertRow(0);

	QIcon ItemIcon;

	switch (Type)
	{
		case (int)QLogger::Normal:
		{
			ItemIcon = GetIcon("information");
			break;
		}

		case (int)QLogger::Critical:
		{
			ItemIcon = GetIcon("exclamation-red");
			break;
		}
	}
	
 	setItem(0, 0, new QTimeTableWidgetItem());
 	setItem(0, 1, new QTableWidgetItem(ItemIcon, ""));
 	setItem(0, 2, new QTableItemMessage(Message, (QLogger::MessageType)Type));
 	setRowHeight(0, 18);
}

void QLogWidget::OnLog(const QString& Message, const QString& Icon)
{
	insertRow(0);

	QIcon ItemIcon = GetIcon(Icon);

	setItem(0, 0, new QTimeTableWidgetItem());
	setItem(0, 1, new QTableWidgetItem(ItemIcon, ""));
	setItem(0, 2, new QTableItemMessage(Message, QLogger::Normal));
	setRowHeight(0, 18);
}

void QLogWidget::OnLogProgress(const QString& Event, const float& Progress)
{
	// Find nearest row with matching event
	QList<QTableWidgetItem*> Items = findItems(Event, Qt::MatchStartsWith);

	int RowIndex = 0;

	if (Items.empty())
	{
		insertRow(0);
		RowIndex = 0;
	}
	else
	{
		RowIndex = Items[0]->row();
	}

	setItem(RowIndex, 0, new QTimeTableWidgetItem());
	setItem(RowIndex, 1, new QTableWidgetItem(""));
	setItem(RowIndex, 2, new QTableItemProgress(Event, Progress));
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
	setRowCount(0);
}

void QLogWidget::contextMenuEvent(QContextMenuEvent* pContextMenuEvent)
{
	QMenu ContextMenu(this);
	ContextMenu.setTitle("Log");

	if (currentRow() > 0)
		ContextMenu.addAction(GetIcon("cross-small"), "Clear", this, SLOT(OnClear()));

	ContextMenu.addAction(GetIcon("cross"), "Clear All", this, SLOT(OnClearAll()));
	ContextMenu.exec(pContextMenuEvent->globalPos());
}