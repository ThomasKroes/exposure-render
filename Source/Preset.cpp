
// Precompiled headers
#include "Stable.h"

#include "Preset.h"

QPresetXML::QPresetXML(QObject* pParent /*= NULL*/, const QString& Name /*= ""*/) :
	QObject(pParent),
	m_Name(Name)
{
}

QPresetXML& QPresetXML::operator=(const QPresetXML& Other)
{
	m_Name = Other.m_Name;

	return *this;
}

QString QPresetXML::GetName(void) const
{
	return m_Name;
}

void QPresetXML::SetName(const QString& Name)
{
	m_Name = Name;
}

void QPresetXML::ReadXML(QDomElement& Parent)
{
	SetName(Parent.attribute("Name"));
}

QDomElement QPresetXML::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	Parent.setAttribute("Name", GetName());
	return Parent;
}