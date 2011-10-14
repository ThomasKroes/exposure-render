
// Precompiled headers
#include "Stable.h"

#include "Film.h"

QFilm::QFilm(QObject* pParent /*= NULL*/) :
	QPresetXML(pParent),
	m_Width(800),
	m_Height(600),
	m_Exposure(0.75f),
	m_Dirty(false)
{
}

QFilm::QFilm(const QFilm& Other)
{
	*this = Other;
}

QFilm& QFilm::operator=(const QFilm& Other)
{
	QPresetXML::operator=(Other);

	m_Width		= Other.m_Width;
	m_Height	= Other.m_Height;
	m_Exposure	= Other.m_Exposure;
	m_Dirty		= Other.m_Dirty;

	emit Changed(*this);

	return *this;
}

int QFilm::GetWidth(void) const
{
	return m_Width;
}

void QFilm::SetWidth(const int& Width)
{
	m_Width	= Width;
	m_Dirty = true;

	emit Changed(*this);
}

int QFilm::GetHeight(void) const
{
	return m_Height;
}

void QFilm::SetHeight(const int& Height)
{
	m_Height	= Height;
	m_Dirty		= true;

	emit Changed(*this);
}

float QFilm::GetExposure(void) const
{
	return m_Exposure;
}

void QFilm::SetExposure(const float& Exposure)
{
	m_Exposure = Exposure;

	emit Changed(*this);
}

bool QFilm::GetNoiseReduction(void) const
{
	return m_NoiseReduction;
}

void QFilm::SetNoiseReduction(const bool& NoiseReduction)
{
	m_NoiseReduction = NoiseReduction;

	emit Changed(*this);
}

void QFilm::Reset(void)
{
	m_Width		= 640;
	m_Height	= 480;
	m_Exposure	= 500.0f;
	m_Dirty		= true;

	emit Changed(*this);
}

bool QFilm::IsDirty(void) const
{
	return m_Dirty;
}

void QFilm::ReadXML(QDomElement& Parent)
{
	m_Width		= Parent.firstChildElement("Width").attribute("Value").toFloat();
	m_Height	= Parent.firstChildElement("Height").attribute("Value").toFloat();
}

QDomElement QFilm::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Film
	QDomElement Film = DOM.createElement("Film");
	Parent.appendChild(Film);

	// Width
	QDomElement Width = DOM.createElement("Width");
	Width.setAttribute("Value", m_Width);
	Film.appendChild(Width);

	// Height
	QDomElement Height = DOM.createElement("Height");
	Height.setAttribute("Value", m_Height);
	Film.appendChild(Height);

	return Film;
}

void QFilm::UnDirty(void)
{
	m_Dirty = false;
}
