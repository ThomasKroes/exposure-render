#pragma once

#include <QtGui>
#include <QtXml\qdom.h>

#include "Utilities.h"
#include "Logger.h"
#include "Status.h"
#include "Controls.h"
#include "CudaUtilities.h"

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

QString GetOpenFileName(const QString& Caption, const QString& Filter, const QString& Icon);
QString GetSaveFileName(const QString& Caption, const QString& Filter, const QString& Icon);
void SaveImage(const unsigned char* pImageBuffer, const int& Width, const int& Height, QString FilePath = "");