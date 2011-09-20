
// Precompiled headers
#include "Stable.h"

#include "MainWindow.h"

#include "VtkWidget.h"
#include "Scene.h"
#include "StartupDialog.h"

// Main window singleton
CMainWindow* gpMainWindow = NULL;

CMainWindow::CMainWindow() :
	QMainWindow(),
	m_CurrentFile(""),
    m_pFileMenu(NULL),
    m_pHelpMenu(NULL),
    m_pFileToolBar(NULL),
	m_pPlaybackToolBar(),
	m_VtkWidget(),
	m_LogDockWidget(),
	m_LightingDockWidget(),
	m_AppearanceDockWidget(),
	m_StatisticsDockWidget(),
	m_CameraDockWidget(),
	m_SettingsDockWidget()
{
	// Set singleton pointer
	gpMainWindow = this;

	// Create VTK rendering window
	setCentralWidget(&m_VtkWidget);

	CreateMenus();
	CreateToolBars();
	CreateStatusBar();
	SetupDockingWidgets();

    setUnifiedTitleAndToolBarOnMac(true);

	setWindowState(Qt::WindowMaximized);

	setWindowFilePath(QString());

	connect(&gRenderStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gRenderStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
}

CMainWindow::~CMainWindow(void)
{
	KillRenderThread();
}

void CMainWindow::CreateMenus(void)
{
    m_pFileMenu = menuBar()->addMenu(tr("&File"));

	m_pFileMenu->addAction(GetIcon("folder-open-document"), "Open", this, SLOT(Open()));

	m_pFileMenu->addSeparator();

	// Recent files actions
	for (int i = 0; i < MaxRecentFiles; ++i)
	{
		m_pRecentFileActions[i] = new QAction(this);
		m_pRecentFileActions[i]->setVisible(false);
		m_pRecentFileActions[i]->setIcon(GetIcon("grid"));
		connect(m_pRecentFileActions[i], SIGNAL(triggered()), this, SLOT(OpenRecentFile()));
		m_pFileMenu->addAction(m_pRecentFileActions[i]);
	}

    m_pFileMenu->addSeparator();

	m_pFileMenu->addAction(GetIcon("star"), "Welcome screen", this, SLOT(ShowStartupDialog()));

	 m_pFileMenu->addSeparator();

	m_pFileMenu->addAction("Close", this, SLOT(Close()));
    m_pFileMenu->addSeparator();
    m_pFileMenu->addAction(GetIcon("door--arrow"), "Exit", this, SLOT(close()));
    
	menuBar()->addSeparator();

	m_pViewMenu = menuBar()->addMenu(tr("&View"));

	
    menuBar()->addSeparator();

    m_pHelpMenu = menuBar()->addMenu(tr("&Help"));
    m_pHelpMenu->addAction(GetIcon("question"), "About Exposure Render", this, SLOT(About()));
	m_pHelpMenu->addAction(GetIcon("question-white"), "About Qt", qApp, SLOT(aboutQt()));


	UpdateRecentFileActions();
}

void CMainWindow::CreateToolBars()
{
	/*
    m_pFileToolBar		= addToolBar(tr("File"));

	m_pFileToolBar->addAction(GetIcon("folder-open"), "Open", this, SLOT(OnOpen()));

	m_pPlaybackToolBar	= addToolBar(tr("Playback"));
	m_pPlaybackToolBar->setIconSize(QSize(16, 16));

	m_pPlaybackToolBar->addAction(GetIcon("control"), "Play", this, SLOT(OnPlay()));
	m_pPlaybackToolBar->addAction(GetIcon("control-pause"), "Pause", this, SLOT(OnPause()));
	m_pPlaybackToolBar->addAction(GetIcon("control-stop-square"), "Stop", this, SLOT(OnStop()));
	m_pPlaybackToolBar->addAction(GetIcon("control-stop-180"), "Restart", this, SLOT(OnRestart()));
	*/
}

void CMainWindow::CreateStatusBar()
{
    statusBar()->showMessage(tr("Ready"));
	statusBar()->setSizeGripEnabled(true);
}

void CMainWindow::SetupDockingWidgets()
{
	// Log dock widget
	m_LogDockWidget.setEnabled(false);
	m_LogDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
	addDockWidget(Qt::RightDockWidgetArea, &m_LogDockWidget);
	m_pViewMenu->addAction(m_LogDockWidget.toggleViewAction());

	// Lighting dock widget
	m_LightingDockWidget.setEnabled(false);
    m_LightingDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::RightDockWidgetArea, &m_LightingDockWidget);
    m_pViewMenu->addAction(m_LightingDockWidget.toggleViewAction());

	// Appearance dock widget
	m_AppearanceDockWidget.setEnabled(false);
	m_AppearanceDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
	addDockWidget(Qt::LeftDockWidgetArea, &m_AppearanceDockWidget);
    m_pViewMenu->addAction(m_AppearanceDockWidget.toggleViewAction());

	// Statistics dock widget
	m_StatisticsDockWidget.setEnabled(false);
	m_StatisticsDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::RightDockWidgetArea, &m_StatisticsDockWidget);
    m_pViewMenu->addAction(m_StatisticsDockWidget.toggleViewAction());

	// Camera dock widget
	m_CameraDockWidget.setEnabled(false);
	m_CameraDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::RightDockWidgetArea, &m_CameraDockWidget);
    m_pViewMenu->addAction(m_CameraDockWidget.toggleViewAction());

	// Settings dock widget
	m_SettingsDockWidget.setEnabled(false);
	m_SettingsDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::LeftDockWidgetArea, &m_SettingsDockWidget);
    m_pViewMenu->addAction(m_SettingsDockWidget.toggleViewAction());

	tabifyDockWidget(&m_AppearanceDockWidget, &m_LightingDockWidget);
	tabifyDockWidget(&m_LightingDockWidget, &m_CameraDockWidget);
//	tabifyDockWidget(&m_CameraDockWidget, &m_SettingsDockWidget);

	m_AppearanceDockWidget.raise();
}

void CMainWindow::UpdateRecentFileActions(void)
{
	QSettings settings;
	QStringList files = settings.value("recentFileList").toStringList();

	int NumRecentFiles = qMin(files.size(), (int)MaxRecentFiles);

	for (int i = 0; i < NumRecentFiles; ++i)
	{
		QString text = tr("&%1 %2").arg(i + 1).arg(StrippedName(files[i]));
		m_pRecentFileActions[i]->setText(text);
		m_pRecentFileActions[i]->setData(files[i]);
		m_pRecentFileActions[i]->setVisible(true);
	}
	
	for (int j = NumRecentFiles; j < MaxRecentFiles; ++j)
		m_pRecentFileActions[j]->setVisible(false);
}

void CMainWindow::OpenRecentFile()
{
	QAction* pAction = qobject_cast<QAction *>(sender());
	
	if (pAction)
		Open(pAction->data().toString());
}

void CMainWindow::SetCurrentFile(const QString& FileName)
 {
     m_CurrentFile = FileName;
     setWindowFilePath(m_CurrentFile);

     QSettings settings;
     QStringList files = settings.value("recentFileList").toStringList();
     files.removeAll(m_CurrentFile);
     files.prepend(FileName);

     while (files.size() > MaxRecentFiles)
         files.removeLast();

     settings.setValue("recentFileList", files);

     foreach (QWidget *widget, QApplication::topLevelWidgets())
	 {
         CMainWindow* pMainWindow = qobject_cast<CMainWindow *>(widget);
         
		 if (pMainWindow)
             pMainWindow->UpdateRecentFileActions();
     }
 }

QString CMainWindow::StrippedName(const QString& FullFileName)
{
	return QFileInfo(FullFileName).fileName();
}

void CMainWindow::Open()
{
	// Create open file dialog
    QString FileName = GetOpenFileName("Open volume", "Meta Image Volume Files (*.mhd)");

	// Exit empty
	if (FileName.isEmpty())
		return;

	// Open the file
	Open(FileName);
}

void CMainWindow::Open(QString FilePath)
{
	// Kill current rendering thread
	KillRenderThread();

	// Window name update
	SetCurrentFile(FilePath);

	// Make string suitable for VTK
	FilePath.replace("/", "\\\\");

 	if (!FilePath.isEmpty())
 		StartRenderThread(FilePath);
}

void CMainWindow::OnLoadDemo(const QString& FileName)
{
	Open(QApplication::applicationDirPath() + "/Examples/" + FileName);
}

void CMainWindow::Close(void)
{
	KillRenderThread();
}

void CMainWindow::Exit(void)
{
	Close();
}

void CMainWindow::About()
{
	QMessageBox::about(this, tr("About Exposure Render"),
		tr("This application illustrates the concepts from the paper: <b>Raytraced lighting in direct volume rendering</b>\n"
		"For more information visit: <b>graphics.tudelft.nl</b>"));

	// Acknowledgments niet vergeten!
}

void CMainWindow::OnRenderBegin(void)
{
	Log("Rendering started", "control");

	for (int i = 0; i < MaxRecentFiles; i++)
	{
		if (m_pRecentFileActions[i]) 
			m_pRecentFileActions[i]->setEnabled(false);
	}
	
	m_LightingDockWidget.setEnabled(true);
	m_AppearanceDockWidget.setEnabled(true);
	m_StatisticsDockWidget.setEnabled(true);
	m_CameraDockWidget.setEnabled(true);
	m_SettingsDockWidget.setEnabled(true);
	m_LogDockWidget.setEnabled(true);
}

void CMainWindow::OnRenderEnd(void)
{
	Log("Rendering ended", "control-stop-square");

	for (int i = 0; i < MaxRecentFiles; i++)
	{
		if (m_pRecentFileActions[i])
			m_pRecentFileActions[i]->setEnabled(true);
	}

	m_LightingDockWidget.setEnabled(false);
	m_AppearanceDockWidget.setEnabled(false);
	m_StatisticsDockWidget.setEnabled(false);
	m_CameraDockWidget.setEnabled(false);
	m_SettingsDockWidget.setEnabled(false);
	m_LogDockWidget.setEnabled(false);
}

void CMainWindow::ShowStartupDialog(void)
{
	QStartupDialog StartupDialog;

	connect(&StartupDialog, SIGNAL(LoadDemo(const QString&)), gpMainWindow, SLOT(OnLoadDemo(const QString&)));

	StartupDialog.exec();
}