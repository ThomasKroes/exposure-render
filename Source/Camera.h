#pragma once

#include "Preset.h"
#include "Film.h"
#include "Aperture.h"
#include "Projection.h"
#include "Focus.h"

class CCamera;

class QCamera : public QPresetXML
{
	Q_OBJECT

public:
	QCamera(QObject* pParent = NULL);
	QCamera::QCamera(const QCamera& Other);
	QCamera& QCamera::operator=(const QCamera& Other);

	QFilm&			GetFilm(void);
	void			SetFilm(const QFilm& Film);
	QAperture&		GetAperture(void);
	void			SetAperture(const QAperture& Aperture);
	QProjection&	GetProjection(void);
	void			SetProjection(const QProjection& Projection);
	QFocus&			GetFocus(void);
	void			SetFocus(const QFocus& Focus);
	void			ReadXML(QDomElement& Parent);
	QDomElement		WriteXML(QDomDocument& DOM, QDomElement& Parent);
	static QCamera	Default(void);

signals:
	void Changed(void);

private:
	QFilm			m_Film;
	QAperture		m_Aperture;
	QProjection		m_Projection;
	QFocus			m_Focus;
};

// Camera singleton
extern QCamera gCamera;