#pragma once

#include <QtGui>

#include "Preset.h"
#include "Background.h"

class QLight : public QPresetXML
{
	Q_OBJECT

public:
	QLight(QObject* pParent = NULL);
	virtual ~QLight(void);

	QLight::QLight(const QLight& Other);
	
	QLight& QLight::operator=(const QLight& Other);

	bool operator == (const QLight& Other) const;

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

class QLighting : public QPresetXML
{
	Q_OBJECT

public:
	QLighting(QObject* pParent = NULL);
	virtual ~QLighting(void);

	QLighting::QLighting(const QLighting& Other);

	QLighting& QLighting::operator=(const QLighting& Other);
	
	void			AddLight(QLight& Light);
	void			RemoveLight(QLight* pLight);
	void			RemoveLight(const int& Index);
	void			CopyLight(QLight* pLight);
	void			CopySelectedLight(void);
	void			RenameLight(const int& Index, const QString& Name);
	QBackground&	Background(void);
	void			SetSelectedLight(QLight* pSelectedLight);
	void			SetSelectedLight(const int& Index);
	QLight*			GetSelectedLight(void);
	void			SelectPreviousLight(void);
	void			SelectNextLight(void);
	void			ReadXML(QDomElement& Parent);
	QDomElement		WriteXML(QDomDocument& DOM, QDomElement& Parent);

	static QLighting Default(void);

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