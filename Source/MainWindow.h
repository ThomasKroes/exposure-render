#pragma once

#include "RenderThread.h"
#include "LightingDockWidget.h"
#include "AppearanceDockWidget.h"
#include "StatisticsDockWidget.h"
#include "CameraDockWidget.h"
#include "SettingsDockWidget.h"
#include "LogDockWidget.h"
#include "VtkWidget.h"

class CMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    CMainWindow(void);
	virtual ~CMainWindow(void);

public slots:
	void Open(void);
	void Open(QString FilePath);
	void OpenRecentFile(void);
	void Close(void);
	void Exit(void);
    void About(void);
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void ShowStartupDialog(void);
	void OnVisitWebsite(void);
	void OnSaveImage(void);

public slots:
	void OnLoadDemo(const QString& FileName);

private:
    void								CreateMenus(void);
    void								CreateToolBars(void);
    void								CreateStatusBar(void);
    void								SetupDockingWidgets(void);
	void								UpdateRecentFileActions(void);
	QString								StrippedName(const QString& FullFileName);
	void								SetCurrentFile(const QString& FileName);

    QString								m_CurrentFile;

public:
	// Menu's
    QMenu*								m_pFileMenu;
    QMenu*								m_pViewMenu;
    QMenu*								m_pHelpMenu;
	CVtkWidget							m_VtkWidget;

private:
	// Toolbars
    QToolBar*							m_pFileToolBar;
	QToolBar*							m_pPlaybackToolBar;

	// Dock widgets
	QLogDockWidget						m_LogDockWidget;
	QLightingDockWidget					m_LightingDockWidget;
	QAppearanceDockWidget				m_AppearanceDockWidget;
	QStatisticsDockWidget				m_StatisticsDockWidget;
	QCameraDockWidget					m_CameraDockWidget;
	QSettingsDockWidget					m_SettingsDockWidget;

	enum
	{
		MaxRecentFiles = 9
	};

    QAction*							m_pRecentFileActions[MaxRecentFiles];
};

extern CMainWindow* gpMainWindow;