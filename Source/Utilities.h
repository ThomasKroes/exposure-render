#pragma once

#include <QtXml\qdom.h>

inline void CreateVectorElement(QDomDocument& DOM, QDomElement& Parent, const QString& Name, const double& X, const double& Y, const double& Z)
{
	QDomElement Vector = DOM.createElement(Name);
	Parent.appendChild(Vector);

	Vector.setAttribute("X", X);
	Vector.setAttribute("Y", Y);
	Vector.setAttribute("Z", Z);
}