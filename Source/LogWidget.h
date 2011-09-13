#pragma once

#include "Controls.h"
#include "Logger.h"

class QLogWidget : public QTableWidget
{
    Q_OBJECT

public:
    QLogWidget(QWidget* pParent = NULL);

	void SetLogger(QLogger* pLogger);

public slots:
	void OnLog(const QString& Message, const QLogger::MessageType& Type);

private:
	QLogger*			m_pLogger;
};