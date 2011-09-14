#pragma once

#include "Controls.h"
#include "Logger.h"

class QTimeTableWidgetItem : public QTableWidgetItem
{
public:
	QTimeTableWidgetItem(void);

	virtual QSize sizeHint() const;
};

class QTableItemMessage : public QTableWidgetItem
{
public:
	QTableItemMessage(const QString& Message, const QLogger::MessageType& MessageType);
};

class QLogWidget : public QTableWidget
{
    Q_OBJECT

public:
    QLogWidget(QWidget* pParent = NULL);

	void SetLogger(QLogger* pLogger);

protected:
	void contextMenuEvent(QContextMenuEvent* pContextMenuEvent);

public slots:
	void OnLog(const QString& Message, const int& Type);
	void OnClear(void);
	void OnClearAll(void);

private:
	QLogger*	m_pLogger;
};