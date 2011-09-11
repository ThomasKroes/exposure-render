#pragma once

#include "Preset.h"

class QBackground : public QPresetXML
{
	Q_OBJECT

public:
	QBackground(QObject* pParent = NULL);
	virtual ~QBackground(void);

	QBackground::QBackground(const QBackground& Other);

	QBackground& QBackground::operator=(const QBackground& Other);

	bool		GetEnabled(void) const;
	void		SetEnabled(const bool& Enable);
	QColor		GetColor(void) const;
	void		SetColor(const QColor& Color);
	float		GetIntensity(void) const;
	void		SetIntensity(const float& Intensity);
	bool		GetUseTexture(void) const;
	void		SetUseTexture(const bool& UseTexture);
	QString		GetFile(void) const;
	void		SetFile(const QString& File);

	void		ReadXML(QDomElement& Parent);
	QDomElement	WriteXML(QDomDocument& DOM, QDomElement& Parent);

signals:
	void BackgroundChanged();

protected:
	bool		m_Enable;
	QColor		m_Color;
	float		m_Intensity;
	bool		m_UseTexture;
	QString		m_File;
};