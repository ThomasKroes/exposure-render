#pragma once

class QPresetXML : public QObject
{
	Q_OBJECT

public:
	QPresetXML(QObject* pParent = NULL, const QString& Name = "");

	QPresetXML::QPresetXML(const QPresetXML& Other)
	{
		m_Name = Other.m_Name;

		*this = Other;
	};

	QPresetXML& operator = (const QPresetXML& Other);

	virtual QString		GetName(void) const;
	virtual void		SetName(const QString& Name);
	virtual void		ReadXML(QDomElement& Parent);
	virtual QDomElement	WriteXML(QDomDocument& DOM, QDomElement& Parent);
	
private:
	QString		m_Name;
};

typedef QList<QPresetXML*> QPresetList;