#pragma once

#include <QtGui>

#include "Preset.h"

class QProjection : public QPresetXML
{
	Q_OBJECT

public:
	QProjection(QObject* pParent = NULL);
	QProjection::QProjection(const QProjection& Other);
	QProjection& QProjection::operator=(const QProjection& Other);

	float			GetFieldOfView(void) const;
	void			SetFieldOfView(const float& FieldOfView);
	void			Reset(void);
	void			ReadXML(QDomElement& Parent);
	QDomElement		WriteXML(QDomDocument& DOM, QDomElement& Parent);

signals:
	void Changed(const QProjection&);

private:
	float				m_FieldOfView;
};