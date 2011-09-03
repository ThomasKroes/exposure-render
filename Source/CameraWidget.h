#pragma once

#include <QtGui>

#include "Preset.h"
#include "FilmWidget.h"
#include "ApertureWidget.h"
#include "ProjectionWidget.h"
#include "FocusWidget.h"
#include "PresetsWidget.h"

class QCamera : public QPresetXML
{
	Q_OBJECT

public:
	QCamera(QObject* pParent = NULL) {};

	QCamera::QCamera(const QCamera& Other)
	{
		*this = Other;
	};

	QCamera& QCamera::operator=(const QCamera& Other)
	{
		QPresetXML::operator=(Other);
		
		return *this;
	}

	void			ReadXML(QDomElement& Parent);
	QDomElement		WriteXML(QDomDocument& DOM, QDomElement& Parent);
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

private:
	QGridLayout					m_MainLayout;
	CFilmWidget					m_FilmWidget;
	CApertureWidget				m_ApertureWidget;
	CProjectionWidget			m_ProjectionWidget;
	CFocusWidget				m_FocusWidget;
	QPresetsWidget<QCamera>		m_PresetsWidget;
};