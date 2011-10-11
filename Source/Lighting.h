#pragma once

#include "Preset.h"
#include "Background.h"
#include "Light.h"

class QLighting : public QPresetXML
{
	Q_OBJECT

public:
	QLighting(QObject* pParent = NULL);
	virtual ~QLighting(void);

	QLighting::QLighting(const QLighting& Other);

	QLighting& QLighting::operator=(const QLighting& Other);
	
	void				AddLight(QLight& Light);
	void				RemoveLight(QLight* pLight);
	void				RemoveLight(const int& Index);
	void				CopyLight(QLight* pLight);
	void				CopySelectedLight(void);
	void				RenameLight(const int& Index, const QString& Name);
	QBackground&		Background(void);
	QLightList&			GetLights(void);
	void				SetSelectedLight(QLight* pSelectedLight);
	void				SetSelectedLight(const int& Index);
	QLight*				GetSelectedLight(void);
	void				SelectPreviousLight(void);
	void				SelectNextLight(void);
	void				ReadXML(QDomElement& Parent);
	QDomElement			WriteXML(QDomDocument& DOM, QDomElement& Parent);
	static QLighting	Default(void);

public slots:
	void OnLightPropertiesChanged(QLight* pLight);
	void OnBackgroundChanged(void);

signals:
	void Changed(void);
	void LightSelectionChanged(QLight*);

protected:
	QLightList		m_Lights;
	QLight*			m_pSelectedLight;
 	QBackground		m_Background;

	friend class QLightsWidget;
	friend class QLightingWidget;
};

// Lighting singleton
extern QLighting gLighting;