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
		return *this;
	}

	void	ReadXML(QDomElement& Parent);
	void	WriteXML(QDomDocument& DOM, QDomElement& Parent);
};