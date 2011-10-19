/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <QtGui>
#include <QtXml\qdom.h>
#include <QHttp>

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