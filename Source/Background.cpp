/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "Background.h"

QBackground::QBackground(QObject* pParent) :
	QPresetXML(pParent),
	m_Enable(true),	
	m_ColorTop(QColor(65, 135, 220)),
	m_ColorMiddle(QColor(65, 135, 220)),
	m_ColorBottom(QColor(65, 135, 220)),
	m_Intensity(150.0f),
	m_UseTexture(false),		
	m_File("")
{
}

QBackground::~QBackground(void)
{
}

QBackground::QBackground(const QBackground& Other)
{
	*this = Other;
};

QBackground& QBackground::operator=(const QBackground& Other)
{
	QPresetXML::operator=(Other);

	m_Enable			= Other.m_Enable;
	m_ColorTop			= Other.m_ColorTop;
	m_ColorMiddle		= Other.m_ColorMiddle;
	m_ColorBottom		= Other.m_ColorBottom;
	m_Intensity			= Other.m_Intensity;
	m_UseTexture		= Other.m_UseTexture;
	m_File				= Other.m_File;

	emit Changed();

	return *this;
}

bool QBackground::GetEnabled(void) const
{
	return m_Enable;
}

void QBackground::SetEnabled(const bool& Enable)
{
	m_Enable = Enable;

	emit Changed();
}

QColor QBackground::GetTopColor(void) const
{
	return m_ColorTop;
}

void QBackground::SetTopColor(const QColor& TopColor)
{
	m_ColorTop = TopColor;

	emit Changed();
}

QColor QBackground::GetMiddleColor(void) const
{
	return m_ColorMiddle;
}

void QBackground::SetMiddleColor(const QColor& MiddleColor)
{
	m_ColorMiddle = MiddleColor;

	emit Changed();
}

QColor QBackground::GetBottomColor(void) const
{
	return m_ColorBottom;
}

void QBackground::SetBottomColor(const QColor& BottomColor)
{
	m_ColorBottom = BottomColor;

	emit Changed();
}

float QBackground::GetIntensity(void) const
{
	return m_Intensity;
}

void QBackground::SetIntensity(const float& Intensity)
{
	m_Intensity = Intensity;

	emit Changed();
}

bool QBackground::GetUseTexture(void) const
{
	return m_UseTexture;
}

void QBackground::SetUseTexture(const bool& Texture)
{
	m_UseTexture = Texture;

	emit Changed();
}

QString QBackground::GetFile(void) const
{
	return m_File;
}

void QBackground::SetFile(const QString& File)
{
	m_File = File;

	emit Changed();
}

void QBackground::ReadXML(QDomElement& Parent)
{
	QPresetXML::ReadXML(Parent);

	SetEnabled(Parent.firstChildElement("Enable").attribute("Value").toInt());
	
	QDomElement TopColor = Parent.firstChildElement("TopColor");
	SetTopColor(QColor(TopColor.attribute("R").toInt(), TopColor.attribute("G").toInt(), TopColor.attribute("B").toInt()));

	QDomElement MiddleColor = Parent.firstChildElement("MiddleColor");
	SetMiddleColor(QColor(MiddleColor.attribute("R").toInt(), MiddleColor.attribute("G").toInt(), MiddleColor.attribute("B").toInt()));

	QDomElement BottomColor = Parent.firstChildElement("BottomColor");
	SetBottomColor(QColor(BottomColor.attribute("R").toInt(), BottomColor.attribute("G").toInt(), BottomColor.attribute("B").toInt()));

	SetIntensity(Parent.firstChildElement("Intensity").attribute("Value").toFloat());

	SetUseTexture(Parent.firstChildElement("UseTexture").attribute("Value").toInt());
	SetFile(Parent.firstChildElement("Height").attribute("Value"));
}

QDomElement QBackground::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Background
	QDomElement Background = DOM.createElement("Background");
	Parent.appendChild(Background);

	// Enable
	QDomElement Enable = DOM.createElement("Enable");
	Enable.setAttribute("Value", m_Enable);
	Background.appendChild(Enable);

	// Top Color
	QDomElement TopColor = DOM.createElement("TopColor");
	TopColor.setAttribute("R", m_ColorTop.red());
	TopColor.setAttribute("G", m_ColorTop.green());
	TopColor.setAttribute("B", m_ColorTop.blue());
	Background.appendChild(TopColor);

	// Middle Color
	QDomElement MiddleColor = DOM.createElement("MiddleColor");
	MiddleColor.setAttribute("R", m_ColorMiddle.red());
	MiddleColor.setAttribute("G", m_ColorMiddle.green());
	MiddleColor.setAttribute("B", m_ColorMiddle.blue());
	Background.appendChild(MiddleColor);

	// Bottom Color
	QDomElement BottomColor = DOM.createElement("BottomColor");
	BottomColor.setAttribute("R", m_ColorBottom.red());
	BottomColor.setAttribute("G", m_ColorBottom.green());
	BottomColor.setAttribute("B", m_ColorBottom.blue());
	Background.appendChild(BottomColor);

	// Intensity
	QDomElement Intensity = DOM.createElement("Intensity");
	Intensity.setAttribute("Value", m_Intensity);
	Background.appendChild(Intensity);

	// Use texture
	QDomElement UseTexture = DOM.createElement("UseTexture");
	UseTexture.setAttribute("Value", m_UseTexture);
	Background.appendChild(UseTexture);

	// File
	QDomElement File = DOM.createElement("File");
	File.setAttribute("Value", m_File);
	Background.appendChild(File);

	return Background;
}