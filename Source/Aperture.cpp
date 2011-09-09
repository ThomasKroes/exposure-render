
#include "Aperture.h"

QAperture::QAperture(QObject* pParent /*= NULL*/) :
	QPresetXML(pParent),
	m_Size(0.05)
{
}

QAperture::QAperture(const QAperture& Other)
{
	*this = Other;
}

QAperture& QAperture::operator=(const QAperture& Other)
{
	QPresetXML::operator=(Other);

	m_Size = Other.m_Size;

	emit Changed();

	return *this;
}

int QAperture::GetSize(void) const
{
	return m_Size;
}

void QAperture::SetSize(const int& Size)
{
	m_Size = Size;

	emit Changed();
}

void QAperture::ReadXML(QDomElement& Parent)
{
	m_Size = Parent.firstChildElement("Size").attribute("Value").toFloat();
}

QDomElement QAperture::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Aperture
	QDomElement Aperture = DOM.createElement("Aperture");
	Parent.appendChild(Aperture);

	// Size
	QDomElement Size = DOM.createElement("Size");
	Size.setAttribute("Value", m_Size);
	Size.appendChild(Size);

	return Aperture;
}