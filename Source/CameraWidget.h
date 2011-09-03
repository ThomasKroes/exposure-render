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
		return *this;
	}

	void	ReadXML(QDomElement& Parent);
	void	WriteXML(QDomDocument& DOM, QDomElement& Parent);
};

class CCameraWidget : public QWidget
{
    Q_OBJECT

public:
    CCameraWidget(QWidget* pParent = NULL);

private:
	QGridLayout					m_MainLayout;
	CFilmWidget					m_FilmWidget;
	CApertureWidget				m_ApertureWidget;
	CProjectionWidget			m_ProjectionWidget;
	CFocusWidget				m_FocusWidget;
	QTemplateWidget<QCamera>	m_PresetsWidget;
};