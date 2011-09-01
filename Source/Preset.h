#pragma once

#include <QtGui>
#include <QtXml\qdom.h>

class QPresetXML : public QObject
{
	Q_OBJECT

public:
	QPresetXML(QObject* pParent = NULL, const QString& Name = "");

	QPresetXML::QPresetXML(const QPresetXML& Other)
	{
		*this = Other;
	};

	QPresetXML& operator = (const QPresetXML& Other);

	virtual void	ReadXML(QDomElement& Parent) = 0;
	virtual void	WriteXML(QDomDocument& DOM, QDomElement& Parent) = 0;
	virtual QString GetName(void) const;
	virtual void	SetName(const QString& Name);

private:
	QString		m_Name;
};

typedef QList<QPresetXML*> QPresetList;