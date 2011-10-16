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

#include "Projection.h"

QProjection::QProjection(QObject* pParent /*= NULL*/) :
	QPresetXML(pParent),
	m_FieldOfView(35.0f)
{
}

QProjection::QProjection(const QProjection& Other)
{
	*this = Other;
}

QProjection& QProjection::operator=(const QProjection& Other)
{
	QPresetXML::operator=(Other);

	m_FieldOfView = Other.m_FieldOfView;

	emit Changed(*this);

	return *this;
}

float QProjection::GetFieldOfView(void) const
{
	return m_FieldOfView;
}

void QProjection::SetFieldOfView(const float& FieldOfView)
{
	m_FieldOfView = FieldOfView;

	emit Changed(*this);
}

void QProjection::Reset(void)
{
	m_FieldOfView = 35.0f;

	emit Changed(*this);
}

void QProjection::ReadXML(QDomElement& Parent)
{
	m_FieldOfView = Parent.firstChildElement("FieldOfView").attribute("Value").toFloat();
}

QDomElement QProjection::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Projection
	QDomElement Projection = DOM.createElement("Projection");
	Parent.appendChild(Projection);

	// Field Of View
	QDomElement FieldOfView = DOM.createElement("FieldOfView");
	FieldOfView.setAttribute("Value", m_FieldOfView);
	Projection.appendChild(FieldOfView);

	return Projection;
}