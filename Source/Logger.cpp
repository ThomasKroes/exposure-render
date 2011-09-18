
// Precompiled headers
#include "Stable.h"

#include "Logger.h"

QLogger gLogger;

void QLogger::SetLog(const QString& Message, const MessageType& Type)
{
	emit Log(Message, Type);
}

void QLogger::SetLog(const QString& Message, const QString& Icon)
{
	emit Log(Message, Icon);
}

void QLogger::SetProgress(const QString& Event, const float& Progress)
{
	emit LogProgress(Event, Progress);
}

void Log(const QString& Message, const QLogger::MessageType& Type /*= QLogger::Normal*/)
{
	gLogger.SetLog(Message, Type);
}

void Log(const QString& Message, const QString& Icon /*= ""*/)
{
	gLogger.SetLog(Message, Icon);
}

void LogProgress(const QString& Event, const float& Progress)
{
	gLogger.SetProgress(Event, Progress);
}