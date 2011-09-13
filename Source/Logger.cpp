
// Precompiled headers
#include "Stable.h"

#include "Logger.h"

QLogger gLogger;

void QLogger::SetLog(const QString& Message, const MessageType& Type)
{
	emit Log(Message, Type);
}

void Log(const QString& Message, const QLogger::MessageType& Type /*= QLogger::Normal*/)
{
	gLogger.SetLog(Message, Type);
}