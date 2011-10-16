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

#include "Camera.h"

QCamera gCamera;

QCamera::QCamera(QObject* pParent /*= NULL*/) :
	QPresetXML(pParent),
	m_Film(),
	m_Aperture(),
	m_Projection(),
	m_Focus(),
	m_From(1.0f),
	m_Target(0.5f),
	m_Up(0.0f, 1.0f, 0.0f)
{
}

QCamera::~QCamera(void)
{
}

QCamera::QCamera(const QCamera& Other)
{
	*this = Other;
};

QCamera& QCamera::operator=(const QCamera& Other)
{
	QPresetXML::operator=(Other);

	blockSignals(true);

	m_Film			= Other.m_Film;
	m_Aperture		= Other.m_Aperture;
	m_Projection	= Other.m_Projection;
	m_Focus			= Other.m_Focus;
	m_From			= Other.m_From;
	m_Target		= Other.m_Target;
	m_Up			= Other.m_Up;

	blockSignals(false);

	emit Changed();

	return *this;
}

QFilm& QCamera::GetFilm(void)
{
	return m_Film;
}

void QCamera::SetFilm(const QFilm& Film)
{
	m_Film = Film;
}

QAperture& QCamera::GetAperture(void)
{
	return m_Aperture;
}

void QCamera::SetAperture(const QAperture& Aperture)
{
	m_Aperture = Aperture;
}

QProjection& QCamera::GetProjection(void)
{
	return m_Projection;
}

void QCamera::SetProjection(const QProjection& Projection)
{
	m_Projection = Projection;
}

QFocus& QCamera::GetFocus(void)
{
	return m_Focus;
}

void QCamera::SetFocus(const QFocus& Focus)
{
	m_Focus = Focus;
}

Vec3f QCamera::GetFrom(void) const
{
	return m_From;
}

void QCamera::SetFrom(const Vec3f& From)
{
	m_From = From;

	emit Changed();
}

Vec3f QCamera::GetTarget(void) const
{
	return m_Target;
}

void QCamera::SetTarget(const Vec3f& Target)
{
	m_Target = Target;

	emit Changed();
}

Vec3f QCamera::GetUp(void) const
{
	return m_Up;
}

void QCamera::SetUp(const Vec3f& Up)
{
	m_Up = Up;

	emit Changed();
}

void QCamera::ReadXML(QDomElement& Parent)
{
	QPresetXML::ReadXML(Parent);

	m_Film.ReadXML(Parent.firstChildElement("Film"));
	m_Aperture.ReadXML(Parent.firstChildElement("Aperture"));
	m_Projection.ReadXML(Parent.firstChildElement("Projection"));
	m_Focus.ReadXML(Parent.firstChildElement("Focus"));

	ReadVectorElement(Parent, "From", m_From.x, m_From.y, m_From.z);
	ReadVectorElement(Parent, "Target", m_Target.x, m_Target.y, m_Target.z);
	ReadVectorElement(Parent, "Up", m_Up.x, m_Up.y, m_Up.z);
}

QDomElement QCamera::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Camera
	QDomElement Camera = DOM.createElement("Preset");
	Parent.appendChild(Camera);

	QPresetXML::WriteXML(DOM, Camera);

	m_Film.WriteXML(DOM, Camera);
	m_Aperture.WriteXML(DOM, Camera);
	m_Projection.WriteXML(DOM, Camera);
	m_Focus.WriteXML(DOM, Camera);

	WriteVectorElement(DOM, Camera, "From", m_From.x, m_From.y, m_From.z);
	WriteVectorElement(DOM, Camera, "Target", m_Target.x, m_Target.y, m_Target.z);
	WriteVectorElement(DOM, Camera, "Up", m_Up.x, m_Up.y, m_Up.z);

	return Camera;
}

QCamera QCamera::Default(void)
{
	QCamera DefaultCamera;

	DefaultCamera.SetName("Default");

	return DefaultCamera;
}

void QCamera::OnFilmChanged(void)
{
	emit Changed();
	SetDirty();
}

void QCamera::OnApertureChanged(void)
{
	emit Changed();
	SetDirty();
}

void QCamera::OnProjectionChanged(void)
{
 	emit Changed();
	SetDirty();
}

void QCamera::OnFocusChanged(void)
{
	emit Changed();
	SetDirty();
}