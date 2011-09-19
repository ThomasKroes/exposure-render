#pragma once

#include <QtGui>
#include <QtXml\qdom.h>

#include "Utilities.h"
#include "Logger.h"
#include "Controls.h"

inline void ReadVectorElement(QDomElement& Parent, const QString& Name, double& X, double& Y, double& Z)
{
	X = Parent.firstChildElement(Name).attribute("X").toDouble();
	Y = Parent.firstChildElement(Name).attribute("X").toDouble();
	Z = Parent.firstChildElement(Name).attribute("X").toDouble();
}

inline void WriteVectorElement(QDomDocument& DOM, QDomElement& Parent, const QString& Name, const double& X, const double& Y, const double& Z)
{
	QDomElement Vector = DOM.createElement(Name);
	Parent.appendChild(Vector);

	Vector.setAttribute("X", X);
	Vector.setAttribute("Y", Y);
	Vector.setAttribute("Z", Z);
}

inline QIcon GetIcon(const QString& Name)
{
	return QIcon(QApplication::applicationDirPath() + "/Icons/" + Name + ".png");
}

QString GetOpenFileName(const QString& Caption, const QString& Filter);

QString GetSaveFileName(const QString& Caption, const QString& Filter);