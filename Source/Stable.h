#pragma once

#include <QtGui>
#include <QtXml\qdom.h>

#include "Logger.h"

inline QIcon GetIcon(const QString& Name)
{
	return QIcon(QApplication::applicationDirPath() + "/Icons/" + Name + ".png");
}