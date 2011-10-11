#pragma once

#include "Preset.h"

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