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

		m_Theta					= Other.m_Theta;
		m_Phi					= Other.m_Phi;
		m_Distance				= Other.m_Distance;
		m_Width					= Other.m_Width;
		m_Height				= Other.m_Height;
		m_LockSize				= Other.m_LockSize;
		m_Color					= Other.m_Color;
		m_Intensity				= Other.m_Intensity;
		
		return *this;
	}

	bool operator == (const QLight& Other) const
	{
		return GetName() == Other.GetName();
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
	void ThetaChanged(QLight*);
	void PhiChanged(QLight*);
	void DistanceChanged(QLight*);
	void WidthChanged(QLight*);
	void HeightChanged(QLight*);
	void LockSizeChanged(QLight*);
	void ColorChanged(QLight*);
	void IntensityChanged(QLight*);

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

class QBackground : public QPresetXML
{
	Q_OBJECT

public:
	QBackground(QObject* pParent = NULL);

	QBackground::QBackground(const QBackground& Other)
	{
		*this = Other;
	};

	QBackground& QBackground::operator=(const QBackground& Other)
	{
		QPresetXML::operator=(Other);

		m_Enable		= Other.m_Enable;
		m_Color			= Other.m_Color;
		m_Intensity		= Other.m_Intensity;
		m_UseTexture	= Other.m_UseTexture;
		m_File			= Other.m_File;

		emit BackgroundChanged();

		return *this;
	}

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

class QLighting : public QPresetXML
{
	Q_OBJECT

public:
	QLighting(QObject* pParent = NULL);;

	QLighting::QLighting(const QLighting& Other)
	{
		*this = Other;
	};

	QLighting& QLighting::operator=(const QLighting& Other)
	{
		QPresetXML::operator=(Other);

		m_Lights		= Other.m_Lights;
		m_Background	= Other.m_Background;

		emit LightingChanged();

		return *this;
	}

	void		ReadXML(QDomElement& Parent);
	QDomElement	WriteXML(QDomDocument& DOM, QDomElement& Parent);
	void AddLight(QLight& Light);
	QBackground& Background(void);
	
	void	SetSelectedLight(QLight* pSelectedLight);
	void	SetSelectedLight(const int& Index);
	QLight*	GetSelectedLight(void);
	void	SelectPreviousLight(void);
	void	SelectNextLight(void);

public slots:
	void OnLightPropertiesChanged(QLight* pLight);
	void Update(void);

signals:
	void LightingChanged(void);
	void LightSelectionChanged(QLight*, QLight*);

protected:
	QLightList		m_Lights;
	QLight*			m_pSelectedLight;
	QBackground		m_Background;

	friend class QLightsWidget;
	friend class QLightingWidget;
};

// Lighting singleton
extern QLighting gLighting;