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

	virtual QString			GetName(void) const;
	virtual void			SetName(const QString& Name);
	virtual bool			GetDirty(void) const;
	virtual void			SetDirty(const bool& Dirty = true);
	virtual void			ReadXML(QDomElement& Parent);
	virtual QDomElement		WriteXML(QDomDocument& DOM, QDomElement& Parent);
	
private:
	QString		m_Name;
	bool		m_Dirty;
};

typedef QList<QPresetXML*> QPresetList;