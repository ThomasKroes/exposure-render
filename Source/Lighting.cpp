
#include "Lighting.h"

QLighting gLighting;

void QLighting::ReadXML(QDomElement& Parent)
{
	QPresetXML::ReadXML(Parent);
}

QDomElement QLighting::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Create transfer function preset root element
	QDomElement Preset = QPresetXML::WriteXML(DOM, Parent);

	return Preset;
}