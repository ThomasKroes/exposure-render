#pragma once

#include <QtGui>
#include <QtXml\qdom.h>

#include "Utilities.h"
#include "Logger.h"
#include "Status.h"
#include "Controls.h"
#include "CudaUtilities.h"

inline void ReadVectorElement(QDomElement& Parent, const QString& Name, float& X, float& Y, float& Z)
{
	X = Parent.firstChildElement(Name).attribute("X").toFloat();
	Y = Parent.firstChildElement(Name).attribute("Y").toFloat();
	Z = Parent.firstChildElement(Name).attribute("Z").toFloat();
}

inline void WriteVectorElement(QDomDocument& DOM, QDomElement& Parent, const QString& Name, const float& X, const float& Y, const float& Z)
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