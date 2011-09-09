#pragma once

#include <QtGui>

#include "Preset.h"

class QAperture : public QPresetXML
{
	Q_OBJECT

public:
	QAperture(QObject* pParent = NULL);
	QAperture::QAperture(const QAperture& Other);
	QAperture& QAperture::operator=(const QAperture& Other);

	int				GetSize(void) const;
	void			SetSize(const int& Size);
	void			ReadXML(QDomElement& Parent);
	QDomElement		WriteXML(QDomDocument& DOM, QDomElement& Parent);

signals:
	void Changed(void);

private:
	float			m_Size;
};