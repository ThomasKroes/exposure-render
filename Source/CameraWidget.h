#pragma once

#include <QtGui>

#include "Preset.h"
#include "Film.h"
#include "Aperture.h"
#include "Projection.h"
#include "Focus.h"
#include "FilmWidget.h"
#include "ApertureWidget.h"
#include "ProjectionWidget.h"
#include "FocusWidget.h"
#include "PresetsWidget.h"

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

	static QCamera Default(void);

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

class CCameraWidget : public QWidget
{
    Q_OBJECT

public:
    CCameraWidget(QWidget* pParent = NULL);

public slots:
	void OnLoadPreset(const QString& Name);
	void OnSavePreset(const QString& Name);
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void Update(void);

private:
	QGridLayout					m_MainLayout;
	CFilmWidget					m_FilmWidget;
	CApertureWidget				m_ApertureWidget;
	CProjectionWidget			m_ProjectionWidget;
	CFocusWidget				m_FocusWidget;
	QPresetsWidget<QCamera>		m_PresetsWidget;
};