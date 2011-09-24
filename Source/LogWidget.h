#pragma once

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

class QTableItemProgress : public QTableWidgetItem
{
public:
	QTableItemProgress(const QString& Event, const float& Progress);
};

class QLogWidget : public QTableWidget
{
    Q_OBJECT

public:
    QLogWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const;

	void SetLogger(QLogger* pLogger);

protected:
	void contextMenuEvent(QContextMenuEvent* pContextMenuEvent);

public slots:
	void OnLog(const QString& Message, const int& Type);
	void OnLog(const QString& Message, const QString& Icon);
	void OnLogProgress(const QString& Event, const float& Progress);
	void OnClear(void);
	void OnClearAll(void);

private:
	QLogger*	m_pLogger;
};