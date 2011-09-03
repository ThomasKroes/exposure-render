#pragma once

#include <QtGui>

#include "Preset.h"

class QLighting : public QPresetXML
{
	Q_OBJECT

public:
	QLighting(QObject* pParent = NULL) {};

	QLighting::QLighting(const QLighting& Other)
	{
		*this = Other;
	};

	QLighting& QLighting::operator=(const QLighting& Other)
	{
		QPresetXML::operator=(Other);

		return *this;
	}

	void		ReadXML(QDomElement& Parent);
	QDomElement	WriteXML(QDomDocument& DOM, QDomElement& Parent);
};

// Lighting singleton
extern QLighting gLighting;