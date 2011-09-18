#pragma once

class QLogger : public QObject
{
	Q_OBJECT

public:
	enum MessageType
	{
		Normal = 0,
		Warning,
		Critical
	};

	void SetLog(const QString& Message, const MessageType& Type);
	void SetLog(const QString& Message, const QString& Icon);
	void SetProgress(const QString& Event, const float& Progress);

signals:
	void Log(const QString& Message, const int& Type);
	void Log(const QString& Message, const QString& Icon);
	void LogProgress(const QString& Event, const float& Progress);
};

extern QLogger gLogger;

void Log(const QString& Message, const QLogger::MessageType& Type = QLogger::Normal);
void Log(const QString& Message, const QString& Icon);
void LogProgress(const QString& Event, const float& Progress);