#pragma once

#include <QtGui>

// VTK
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkConeSource.h>
#include <vtkRenderWindow.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkImageImport.h>
#include <vtkImageActor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkVolumeMapper.h>

// Interactor
#include "InteractorStyleRealisticCamera.h"

// Render thread
#include "RenderThread.h"

// Render thread
extern CRenderThread* gpRenderThread;

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
	void Open();
	void OpenRecentFile();
	void Close();
	void Exit();
    void About();
	void OnTimer();

private:
    void								CreateActions(void);
    void								CreateMenus(void);
    void								CreateToolBars(void);
    void								CreateStatusBar(void);
    void								CreateDockWindows(void);
	void								UpdateRecentFileActions(void);
	QString								StrippedName(const QString& FullFileName);
	void								SetCurrentFile(const QString& FileName);
	void								LoadFile(const QString& FileName);
	void								SetupRenderView(void);
    QString								m_CurrentFile;

public:
	vtkImageImport*						m_pImageImport;
	vtkImageActor*						m_pImageActor;
	vtkInteractorStyleImage*			m_pInteractorStyleImage;
	vtkRenderer*						m_pSceneRenderer;
	vtkRenderWindow*					m_pRenderWindow;
	vtkRenderWindowInteractor*			m_pRenderWindowInteractor;
	vtkCallbackCommand*					m_pTimerCallback;
	vtkCallbackCommand*					m_pKeyPressCallback;
	vtkCallbackCommand*					m_pKeyReleaseCallback;
	CInteractorStyleRealisticCamera*	m_pInteractorStyleRealisticCamera;

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
	QDockWidget*						m_pLightingDockWidget;
	QDockWidget*						m_pVolumeAppearanceDockWidget;
	QDockWidget*						m_pStatisticsDockWidget;
	QDockWidget*						m_pCameraDockWidget;
	QDockWidget*						m_pSettingsDockWidget;

	// Actions
    QAction*							m_pOpenAct;
	QAction*							m_pCloseAct;
    QAction*							m_pExitAct;
    QAction*							m_pAboutAct;
    QAction*							m_pAboutQtAct;
	QAction*							m_pSeparatorAction;

	// Timers
	QTimer								m_Timer;

	enum
	{
		MaxRecentFiles = 9
	};

    QAction*							m_pRecentFileActions[MaxRecentFiles];
};

extern CMainWindow* gpMainWindow;