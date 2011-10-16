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

#include "Preset.h"

class QBackground : public QPresetXML
{
	Q_OBJECT

public:
	QBackground(QObject* pParent = NULL);
	QBackground::QBackground(const QBackground& Other);
	virtual ~QBackground(void);
	QBackground& QBackground::operator=(const QBackground& Other);

	bool		GetEnabled(void) const;
	void		SetEnabled(const bool& Enable);
	QColor		GetTopColor(void) const;
	void		SetTopColor(const QColor& TopColor);
	QColor		GetMiddleColor(void) const;
	void		SetMiddleColor(const QColor& MiddleColor);
	QColor		GetBottomColor(void) const;
	void		SetBottomColor(const QColor& BottomColor);
	float		GetIntensity(void) const;
	void		SetIntensity(const float& Intensity);
	bool		GetUseTexture(void) const;
	void		SetUseTexture(const bool& UseTexture);
	QString		GetFile(void) const;
	void		SetFile(const QString& File);

	void		ReadXML(QDomElement& Parent);
	QDomElement	WriteXML(QDomDocument& DOM, QDomElement& Parent);

signals:
	void Changed();

protected:
	bool		m_Enable;
	QColor		m_ColorTop;
	QColor		m_ColorMiddle;
	QColor		m_ColorBottom;
	float		m_Intensity;
	bool		m_UseTexture;
	QString		m_File;
};