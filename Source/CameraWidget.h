#pragma once

#include <QtGui>

#include "Camera.h"
#include "FilmWidget.h"
#include "ApertureWidget.h"
#include "ProjectionWidget.h"
#include "FocusWidget.h"
#include "PresetsWidget.h"

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