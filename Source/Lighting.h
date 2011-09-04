#pragma once

#include <QtGui>

#include "Preset.h"

class QLight : public QPresetXML
{
	Q_OBJECT

public:
	QLight(QObject* pParent = NULL);

	QLight::QLight(const QLight& Other)
	{
		*this = Other;
	};

	QLight& QLight::operator=(const QLight& Other)
	{
		QPresetXML::operator=(Other);

		m_Theta			= Other.m_Theta;
		m_Phi			= Other.m_Phi;
		m_Distance		= Other.m_Distance;
		m_Width			= Other.m_Width;
		m_Height		= Other.m_Height;
		m_LockSize		= Other.m_LockSize;
		m_Color			= Other.m_Color;
		m_Intensity		= Other.m_Intensity;

		return *this;
	}

	float		GetTheta(void) const;
	void		SetTheta(const float& Theta);
	float		GetPhi(void) const;
	void		SetPhi(const float& Phi);
	float		GetWidth(void) const;
	void		SetWidth(const float& Width);
	float		GetHeight(void) const;
	void		SetHeight(const float& Height);
	bool		GetLockSize(void) const;
	void		SetLockSize(const bool& LockSize);
	float		GetDistance(void) const;
	void		SetDistance(const float& Distance);
	QColor		GetColor(void) const;
	void		SetColor(const QColor& Color);
	float		GetIntensity(void) const;
	void		SetIntensity(const float& Intensity);

	void		ReadXML(QDomElement& Parent);
	QDomElement	WriteXML(QDomDocument& DOM, QDomElement& Parent);

signals:
	void LightPropertiesChanged(QLight*);

protected:
	float		m_Theta;
	float		m_Phi;
	float		m_Distance;
	float		m_Width;
	float		m_Height;
	bool		m_LockSize;
	QColor		m_Color;
	float		m_Intensity;

	friend class QLightItem;
};

typedef QList<QLight> QLightList;

class QLighting : public QPresetXML
{
	Q_OBJECT

public:
	QLighting(QObject* pParent = NULL);;

	QLighting::QLighting(const QLighting& Other)
	{
		m_Lights  = Other.m_Lights;

		*this = Other;
	};

	QLighting& QLighting::operator=(const QLighting& Other)
	{
		QPresetXML::operator=(Other);

		m_Lights = Other.m_Lights;

		emit LightingChanged();

		return *this;
	}

	void		ReadXML(QDomElement& Parent);
	QDomElement	WriteXML(QDomDocument& DOM, QDomElement& Parent);
	void AddLight(QLight& Light);

public slots:
	void OnLightPropertiesChanged(QLight* pLight);

signals:
	void LightingChanged(void);

protected:
	QLightList	m_Lights;

	friend class QLightsWidget;
};

// Lighting singleton
extern QLighting gLighting;