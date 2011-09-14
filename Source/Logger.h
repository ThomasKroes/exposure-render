#pragma once

class QLogger : public QObject
{
	Q_OBJECT

public:
	enum MessageType
	{
		Normal		= 0x0001,
		Warning		= 0x0002,
		Critical	= 0x0004
	};

	void SetLog(const QString& Message, const MessageType& Type);
	void SetProgress(const QString& Event, const float& Progress);

signals:
	void Log(const QString& Message, const int& Type);
	void LogProgress(const QString& Event, const float& Progress);
};

extern QLogger gLogger;

void Log(const QString& Message, const QLogger::MessageType& Type = QLogger::Normal);
void LogProgress(const QString& Event, const float& Progress);