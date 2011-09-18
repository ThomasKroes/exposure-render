
// Precompiled headers
#include "Stable.h"

#include "Camera.h"

QCamera gCamera;

QCamera::QCamera(QObject* pParent /*= NULL*/) :
	QPresetXML(pParent),
	m_Film(),
	m_Aperture(),
	m_Projection(),
	m_Focus()
{
	QObject::connect(&m_Film, SIGNAL(Changed(const QFilm&)), this, SLOT(OnFilmChanged()));
	QObject::connect(&m_Aperture, SIGNAL(Changed(const QAperture&)), this, SLOT(OnApertureChanged()));
	QObject::connect(&m_Projection, SIGNAL(Changed(const QProjection&)), this, SLOT(OnProjectionChanged()));
	QObject::connect(&m_Focus, SIGNAL(Changed(const QFocus&)), this, SLOT(OnFocusChanged()));
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

void QCamera::ReadXML(QDomElement& Parent)
{
	QPresetXML::ReadXML(Parent);

	m_Film.ReadXML(Parent.firstChildElement("Film"));
	m_Aperture.ReadXML(Parent.firstChildElement("Aperture"));
	m_Projection.ReadXML(Parent.firstChildElement("Projection"));
	m_Focus.ReadXML(Parent.firstChildElement("Focus"));
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
}

void QCamera::OnApertureChanged(void)
{
	emit Changed();
}

void QCamera::OnProjectionChanged(void)
{
	emit Changed();
}

void QCamera::OnFocusChanged(void)
{
	emit Changed();
}