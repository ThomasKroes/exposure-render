#pragma once

#include <QtGui>

#include "Logger.h"
#include "Controls.h"
#include "Utilities.h"

inline QIcon GetIcon(const QString& Name)
{
	return QIcon(QApplication::applicationDirPath() + "/Icons/" + Name + ".png");
}