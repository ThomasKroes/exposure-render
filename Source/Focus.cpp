
#include "Focus.h"

QFocus::QFocus(QObject* pParent /*= NULL*/) :
	QPresetXML(pParent),
	m_FocalDistance(1.0f)
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

	emit Changed();

	return *this;
}

int QFocus::GetFocalDistance(void) const
{
	return m_FocalDistance;
}

void QFocus::SetFocalDistance(const int& FocalDistance)
{
	m_FocalDistance = FocalDistance;

	emit Changed();
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