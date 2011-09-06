#pragma once

// Qt
#include <QtGui>

// Render thread
#include "RenderThread.h"

// Dock widgets
#include "LightingDockWidget.h"
#include "AppearanceDockWidget.h"
#include "StatisticsDockWidget.h"
#include "CameraDockWidget.h"
#include "SettingsDockWidget.h"

class QAction;
class QListWidget;
class QMenu;
class QTextEdit;
class CVtkWidget;

class CMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    CMainWindow(void);
	virtual ~CMainWindow(void);

private slots:
	void Open(void);
	void Open(QString FilePath);
	void OpenRecentFile(void);
	void Close(void);
	void Exit(void);
    void About(void);
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	
private:
    void								CreateActions(void);
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

private:
	// Toolbars
    QToolBar*							m_pFileToolBar;
    
	// Qt-VTK widget					
	CVtkWidget*							m_pVtkWidget;

	// Dock widgets
	QLightingDockWidget					m_LightingDockWidget;
	QAppearanceDockWidget				m_AppearanceDockWidget;
	QStatisticsDockWidget				m_StatisticsDockWidget;
	QCameraDockWidget					m_CameraDockWidget;
	QSettingsDockWidget					m_SettingsDockWidget;

	// Actions
    QAction*							m_pOpenAct;
	QAction*							m_pCloseAct;
    QAction*							m_pExitAct;
    QAction*							m_pAboutAct;
    QAction*							m_pAboutQtAct;
	QAction*							m_pSeparatorAction;

	enum
	{
		MaxRecentFiles = 9
	};

    QAction*							m_pRecentFileActions[MaxRecentFiles];
};

extern CMainWindow* gpMainWindow;