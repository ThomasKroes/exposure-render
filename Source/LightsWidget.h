#pragma once

#include <QtGui>

class QLight : public QObject
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
		m_Name = Other.m_Name;

		return *this;
	}

	QString		GetName(void);
	void		SetName(const QString& Name);
	float		GetTheta(void) const;
	void		SetTheta(const float& Theta);
	float		GetPhi(void) const;
	void		SetPhi(const float& Phi);

	float		GetWidth(void) const;
	void		SetWidth(const float& Width);

	float		GetHeight(void) const;
	void		SetHeight(const float& Height);

	float		GetDistance(void) const;
	void		SetDistance(const float& Distance);
	QColor		GetColor(void) const;
	void		SetColor(const QColor& Color);
	float		GetIntensity(void) const;
	void		SetIntensity(const float& Intensity);

signals:
	void LightPropertiesChanged(QLight*);

protected:
	QString		m_Name;
	float		m_Theta;
	float		m_Phi;
	float		m_Distance;
	float		m_Width;
	float		m_Height;
	QColor		m_Color;
	float		m_Intensity;

	friend class QLightItem;
};

typedef QList<QLight> QLightList;

class QLightItem : public QListWidgetItem
{
public:
	QLightItem(QListWidget* pListWidget, QLight* pLight) :
		QListWidgetItem(pListWidget),
		m_pLight(pLight)
	  {
		  setText(pLight->m_Name);
	  }

	  QLight*	m_pLight;
};

class QLightsWidget : public QGroupBox
{
    Q_OBJECT

public:
    QLightsWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const { return QSize(200, 200); }

protected slots:
	void UpdateLightList(void);
	void OnLightSelectionChanged(void);
	void OnPresetNameChanged(const QString& Text);
	void OnLightItemChanged(QListWidgetItem* pWidgetItem);
	void OnAddLight(void);
	void OnRemoveLight(void);
	void OnLightPropertiesChanged(QLight* pLight);

signals:
	void LightSelectionChanged(QLight*);
	void LightCountChanged(void);

protected:
	QGridLayout		m_MainLayout;
	QLightList		m_Lights;
	QListWidget		m_LightList;
	QLineEdit		m_LightName;
	QPushButton		m_AddLight;
	QPushButton		m_RemoveLight;

	friend class QLightingWidget;
};