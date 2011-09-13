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
	QColor		GetTopColor(void) const;
	void		SetTopColor(const QColor& TopColor);
	QColor		GetMiddleColor(void) const;
	void		SetMiddleColor(const QColor& MiddleColor);
	QColor		GetBottomColor(void) const;
	void		SetBottomColor(const QColor& BottomColor);
	float		GetTopIntensity(void) const;
	void		SetTopIntensity(const float& TopIntensity);
	float		GetMiddleIntensity(void) const;
	void		SetMiddleIntensity(const float& MiddleIntensity);
	float		GetBottomIntensity(void) const;
	void		SetBottomIntensity(const float& BottomIntensity);
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
	QColor		m_ColorTop;
	QColor		m_ColorMiddle;
	QColor		m_ColorBottom;
	float		m_IntensityTop;
	float		m_IntensityMiddle;
	float		m_IntensityBottom;
	bool		m_UseTexture;
	QString		m_File;
};