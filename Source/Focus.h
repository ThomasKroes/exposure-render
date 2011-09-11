#pragma once

#include "Preset.h"

class QFocus : public QPresetXML
{
	Q_OBJECT

public:
	QFocus(QObject* pParent = NULL);
	QFocus::QFocus(const QFocus& Other);
	QFocus& QFocus::operator=(const QFocus& Other);

	float		GetFocalDistance(void) const;
	void		SetFocalDistance(const float& FocalDistance);
	void		Reset(void);
	void		ReadXML(QDomElement& Parent);
	QDomElement	WriteXML(QDomDocument& DOM, QDomElement& Parent);

signals:
	void Changed(const QFocus&);

private:
	float		m_FocalDistance;
};
