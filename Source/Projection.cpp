
#include "Projection.h"

QProjection::QProjection(QObject* pParent /*= NULL*/) :
	QPresetXML(pParent),
	m_FieldOfView(35)
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

int QProjection::GetFieldOfView(void) const
{
	return m_FieldOfView;
}

void QProjection::SetFieldOfView(const int& FieldOfView)
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