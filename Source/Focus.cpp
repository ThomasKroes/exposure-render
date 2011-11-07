/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "Focus.h"

QFocus::QFocus(QObject* pParent /*= NULL*/) :
	QPresetXML(pParent),
	m_FocalDistance(0.75f)
{
}

QFocus::QFocus(const QFocus& Other)
{
	*this = Other;
}

QFocus& QFocus::operator=(const QFocus& Other)
{
	QPresetXML::operator=(Other);

	m_FocalDistance = Other.m_FocalDistance;

	emit Changed(*this);

	return *this;
}

int QFocus::GetType(void) const
{
	return m_Type;
}

void QFocus::SetType(const int& Type)
{
	m_Type = Type;

	emit Changed(*this);
}

float QFocus::GetFocalDistance(void) const
{
	return m_FocalDistance;
}

void QFocus::SetFocalDistance(const float& FocalDistance)
{
	m_FocalDistance = FocalDistance;

	emit Changed(*this);
}

void QFocus::Reset(void)
{
	m_FocalDistance = 1.0f;

	emit Changed(*this);
}

void QFocus::ReadXML(QDomElement& Parent)
{
	m_FocalDistance = Parent.firstChildElement("FocalDistance").attribute("Value").toFloat();
}

QDomElement QFocus::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Focus
	QDomElement Focus = DOM.createElement("Focus");
	Parent.appendChild(Focus);

	// Focal Distance
	QDomElement FocalDistance = DOM.createElement("FocalDistance");
	FocalDistance.setAttribute("Value", m_FocalDistance);
	Focus.appendChild(FocalDistance);

	return Focus;
}