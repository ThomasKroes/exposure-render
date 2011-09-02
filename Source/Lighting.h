#pragma once

#include <QtGui>

class QLighting : public QObject
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
};