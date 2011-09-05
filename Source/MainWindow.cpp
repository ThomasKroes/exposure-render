
#include "MainWindow.h"

#include "VtkWidget.h"
#include "LoadVolume.h"
#include "Scene.h"

CRenderThread* gpRenderThread = NULL;

// Main window singleton
CMainWindow* gpMainWindow = NULL;

// Timer callback
void TimerCallbackFunction(vtkObject* pCaller, long unsigned int EventId, void* pClientData, void* pCallData)
{
}

// Key press callback
void KeyPressCallbackFunction(vtkObject* pCaller, long unsigned int EventId, void* pClientData, void* pCallData)
{
	vtkRenderWindowInteractor* pRenderWindowInteractor = static_cast<vtkRenderWindowInteractor*>(pCaller);

	char* pKeySymbol = pRenderWindowInteractor->GetKeySym();

	if (strcmp(pKeySymbol, "space") == 0)
	{
		pRenderWindowInteractor->SetInteractorStyle(gpMainWindow->m_pInteractorStyleImage);

		// Change the cursor to a pointing, thus indicating the change in interaction mode
		gpMainWindow->setCursor(QCursor(Qt::CursorShape::PointingHandCursor));
	}
}

// Key press callback
void KeyReleaseCallbackFunction(vtkObject* pCaller, long unsigned int EventId, void* pClientData, void* pCallData)
{
	vtkRenderWindowInteractor* pRenderWindowInteractor = static_cast<vtkRenderWindowInteractor*>(pCaller);

	char* pKeySymbol = pRenderWindowInteractor->GetKeySym();

	if (strcmp(pKeySymbol, "space") == 0)
	{
		pRenderWindowInteractor->SetInteractorStyle(gpMainWindow->m_pInteractorStyleRealisticCamera);

		// Change the cursor to a pointing, thus indicating the change in interaction mode
		gpMainWindow->setCursor(QCursor(Qt::CursorShape::ArrowCursor));
	}
}

CMainWindow::CMainWindow() :
	m_CurrentFile(""),
	m_pImageImport(NULL),
	m_pImageActor(NULL),
	m_pInteractorStyleImage(NULL),
	m_pSceneRenderer(NULL),
	m_pRenderWindow(NULL),
	m_pRenderWindowInteractor(NULL),
	m_pTimerCallback(NULL),
	m_pKeyPressCallback(NULL),
	m_pKeyReleaseCallback(NULL),
	m_pInteractorStyleRealisticCamera(NULL),
    m_pFileMenu(NULL),
    m_pHelpMenu(NULL),
    m_pFileToolBar(NULL),
	m_pVtkWidget(NULL),
	m_LightingDockWidget(),
	m_AppearanceDockWidget(),
	m_StatisticsDockWidget(),
	m_CameraDockWidget(),
	m_SettingsDockWidget(),
    m_pOpenAct(NULL),
	m_pCloseAct(NULL),
    m_pExitAct(NULL),
    m_pAboutAct(NULL),
    m_pAboutQtAct(NULL),
	m_pSeparatorAction(NULL),
	m_Timer()
{
	// Set singleton pointer
	gpMainWindow = this;

	// Create VTK rendering window
	m_pVtkWidget = new CVtkWidget();
	setCentralWidget(m_pVtkWidget);

	CreateActions();
	CreateMenus();
	CreateToolBars();
	CreateStatusBar();
	SetupDockingWidgets();

    setUnifiedTitleAndToolBarOnMac(true);

	// Resize the window
	resize(1280, 768);

	// Setup the VTK render view
	SetupRenderView();

	setWindowFilePath(QString());

	connect(&m_Timer, SIGNAL(timeout()), this, SLOT(OnTimer()));
}

CMainWindow::~CMainWindow(void)
{
	emit CloseRenderThread();
}

void CMainWindow::CreateActions(void)
{
	// Open action
    m_pOpenAct = new QAction(QIcon(":/images/open.png"), tr("&Open..."), this);
    m_pOpenAct->setShortcuts(QKeySequence::Open);
    m_pOpenAct->setStatusTip(tr("Open an existing file"));
    connect(m_pOpenAct, SIGNAL(triggered()), this, SLOT(Open()));

	// Recent files actions
	for (int i = 0; i < MaxRecentFiles; ++i)
	{
         m_pRecentFileActions[i] = new QAction(this);
         m_pRecentFileActions[i]->setVisible(false);
         connect(m_pRecentFileActions[i], SIGNAL(triggered()), this, SLOT(OpenRecentFile()));
     }

	// Close action
	m_pCloseAct = new QAction(QIcon(":/images/open.png"), tr("&Close..."), this);
    m_pCloseAct->setShortcuts(QKeySequence::Close);
    m_pCloseAct->setStatusTip(tr("Close current file"));
    connect(m_pCloseAct, SIGNAL(triggered()), this, SLOT(Close()));

	// Exit action
    m_pExitAct = new QAction(tr("E&xit"), this);
    m_pExitAct->setShortcuts(QKeySequence::Quit);
    m_pExitAct->setStatusTip(tr("Exit the application"));
    connect(m_pExitAct, SIGNAL(triggered()), this, SLOT(close()));

	// About this application action
    m_pAboutAct = new QAction(tr("&About"), this);
    m_pAboutAct->setStatusTip(tr("Show the application's About box"));
    connect(m_pAboutAct, SIGNAL(triggered()), this, SLOT(About()));

	// About Qt action
    m_pAboutQtAct = new QAction(tr("About &Qt"), this);
    m_pAboutQtAct->setStatusTip(tr("Show the Qt library's About box"));
    connect(m_pAboutQtAct, SIGNAL(triggered()), qApp, SLOT(aboutQt()));
}

void CMainWindow::CreateMenus(void)
{
    m_pFileMenu = menuBar()->addMenu(tr("&File"));
    m_pFileMenu->addAction(m_pOpenAct);

	// Separator
	m_pSeparatorAction = m_pFileMenu->addSeparator();

     for (int i = 0; i < MaxRecentFiles; ++i)
         m_pFileMenu->addAction(m_pRecentFileActions[i]);

    m_pFileMenu->addSeparator();

	m_pFileMenu->addAction(m_pCloseAct);
    m_pFileMenu->addSeparator();
    m_pFileMenu->addAction(m_pExitAct);
    
	menuBar()->addSeparator();

	m_pViewMenu = menuBar()->addMenu(tr("&View"));

    menuBar()->addSeparator();

    m_pHelpMenu = menuBar()->addMenu(tr("&Help"));
    m_pHelpMenu->addAction(m_pAboutAct);
    m_pHelpMenu->addAction(m_pAboutQtAct);

	UpdateRecentFileActions();
}

void CMainWindow::CreateToolBars()
{
    m_pFileToolBar = addToolBar(tr("File"));
}

void CMainWindow::CreateStatusBar()
{
    statusBar()->showMessage(tr("Ready"));
	statusBar()->setSizeGripEnabled(true);
}

void CMainWindow::SetupDockingWidgets()
{
	/*
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
	m_StatisticsDockWidget.Init();
	m_StatisticsDockWidget.setEnabled(false);
	m_StatisticsDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::LeftDockWidgetArea, &m_StatisticsDockWidget);
    m_pViewMenu->addAction(m_StatisticsDockWidget.toggleViewAction());

	// Camera dock widget
	m_CameraDockWidget.setEnabled(false);
	m_CameraDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::RightDockWidgetArea, &m_CameraDockWidget);
    m_pViewMenu->addAction(m_CameraDockWidget.toggleViewAction());

	// Settings dock widget
	m_SettingsDockWidget.setEnabled(false);
	m_SettingsDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::RightDockWidgetArea, &m_SettingsDockWidget);
    m_pViewMenu->addAction(m_SettingsDockWidget.toggleViewAction());

	tabifyDockWidget(&m_AppearanceDockWidget, &m_LightingDockWidget);
	tabifyDockWidget(&m_LightingDockWidget, &m_CameraDockWidget);
	tabifyDockWidget(&m_CameraDockWidget, &m_SettingsDockWidget);

	m_AppearanceDockWidget.raise();
	*/
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

	m_pSeparatorAction->setVisible(NumRecentFiles > 0);
}

void CMainWindow::OpenRecentFile()
{
	QAction* pAction = qobject_cast<QAction *>(sender());
	
	if (pAction)
		LoadFile(pAction->data().toString());
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

	SetCurrentFile(FileName);

	// Make string suitable for VTK
	FileName.replace("/", "\\\\");

	if (!FileName.isEmpty())
		LoadFile(FileName);
}

void CMainWindow::LoadFile(const QString& FileName)
{
	// Status message
	statusBar()->showMessage(tr("File loaded"), 2000);

	// Create new volume
	gpScene = new CScene;

	// Create new render thread
	gpRenderThread = new CRenderThread(FileName, this);

	// Load the VTK volume
	if (!LoadVtkVolume(FileName.toAscii().data(), gpScene, gpRenderThread->m_pImageDataVolume))
		return;

	// Force the render thread to allocate the necessary buffers, do not remove this line
	gpScene->m_DirtyFlags.SetFlag(FilmResolutionDirty | CameraDirty);

	gpRenderThread->start();

	// Let us know when the rendering begins and ends
	connect(gpRenderThread, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(gpRenderThread, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	connect(this, SIGNAL(CloseRenderThread()), gpRenderThread, SLOT(OnCloseRenderThread()));
}

void CMainWindow::SetupRenderView(void)
{
	// Create and configure scene renderer
	m_pSceneRenderer = vtkRenderer::New();
	m_pSceneRenderer->SetBackground(0.4, 0.4, 0.43);
	m_pSceneRenderer->SetBackground2(0.6, 0.6, 0.6);
	m_pSceneRenderer->SetGradientBackground(true);
	m_pSceneRenderer->GetActiveCamera()->SetPosition(0.0, 0.0, 1.0);
	m_pSceneRenderer->GetActiveCamera()->SetFocalPoint(0.0, 0.0, 0.0);
	m_pSceneRenderer->GetActiveCamera()->ParallelProjectionOn();
	
	// Get render window and configure
	m_pRenderWindow = m_pVtkWidget->GetQtVtkWidget()->GetRenderWindow();
	m_pRenderWindow->AddRenderer(m_pSceneRenderer);

	// Timer callback
	m_pTimerCallback = vtkCallbackCommand::New();
	m_pTimerCallback->SetCallback(TimerCallbackFunction);

	// Key press callback
	m_pKeyPressCallback = vtkCallbackCommand::New();
	m_pKeyPressCallback->SetCallback(KeyPressCallbackFunction);

	// Key press callback
	m_pKeyReleaseCallback = vtkCallbackCommand::New();
	m_pKeyReleaseCallback->SetCallback(KeyReleaseCallbackFunction);

	// Create interactor style for camera navigation
	m_pInteractorStyleRealisticCamera = CInteractorStyleRealisticCamera::New();
	m_pInteractorStyleImage = vtkInteractorStyleImage::New();

	// Add observers
	m_pRenderWindow->GetInteractor()->SetInteractorStyle(m_pInteractorStyleRealisticCamera);
//	m_pRenderWindow->GetInteractor()->SetInteractorStyle(m_pInteractorStyleImage);

	m_pRenderWindow->GetInteractor()->AddObserver(vtkCommand::TimerEvent, m_pTimerCallback);
	m_pRenderWindow->GetInteractor()->AddObserver(vtkCommand::KeyPressEvent, m_pKeyPressCallback);
	m_pRenderWindow->GetInteractor()->AddObserver(vtkCommand::KeyReleaseEvent, m_pKeyReleaseCallback);
}

void CMainWindow::OnTimer(void)
{
	if (!gpRenderThread || !gpRenderThread->RenderImage() | !gpScene)
		return;
	
	m_pImageImport->SetDataExtent(0, gpScene->m_Camera.m_Film.m_Resolution.Width() - 1, 0, gpScene->m_Camera.m_Film.m_Resolution.Height() - 1, 0, 0);
	m_pImageImport->SetWholeExtent(0, gpScene->m_Camera.m_Film.m_Resolution.Width() - 1, 0, gpScene->m_Camera.m_Film.m_Resolution.Height() - 1, 0, 0);
	m_pImageActor->SetDisplayExtent(0, gpScene->m_Camera.m_Film.m_Resolution.Width() - 1, 0, gpScene->m_Camera.m_Film.m_Resolution.Height() - 1, 0, 0);

//	m_pImageImport->setup(0, gpScene->m_Camera.m_Film.m_Resolution.Width() - 1, 0, gpScene->m_Camera.m_Film.m_Resolution.Height() - 1, 0, 0);
	m_pImageImport->Update();
	m_pImageImport->SetImportVoidPointer(NULL);
	m_pImageImport->SetImportVoidPointer(gpRenderThread->RenderImage());
 	
	m_pImageActor->SetInput(gpMainWindow->m_pImageImport->GetOutput());
// 	m_pImageActor->VisibilityOn();

	m_pRenderWindow->GetInteractor()->Render();
}

void CMainWindow::Close()
{
	emit CloseRenderThread();
}

void CMainWindow::Exit()
{
	
}

void CMainWindow::About()
{
	QMessageBox::about(this, tr("About Exposure Render"),
		tr("This application illustrates the concepts from the paper: <b>Raytraced lighting in direct volume rendering</b>\n"
		"For more information visit: <b>graphics.tudelft.nl</b>"));
}

void CMainWindow::OnRenderBegin(void)
{
	qDebug("Rendering started");

	m_LightingDockWidget.setEnabled(true);
	m_AppearanceDockWidget.setEnabled(true);
	m_StatisticsDockWidget.setEnabled(true);
	m_CameraDockWidget.setEnabled(true);
	m_SettingsDockWidget.setEnabled(true);

//	CResolution2D FilmResolution = gpScene->m_Camera.m_Film.m_Resolution;

	// Create and configure image importer
	m_pImageImport = vtkImageImport::New();
	m_pImageImport->SetDataSpacing(1, 1, 1);
	m_pImageImport->SetDataOrigin(-400, -300, 0);
	m_pImageImport->SetImportVoidPointer((void*)malloc(3 * 800 * 600 * sizeof(unsigned char)));
	m_pImageImport->SetWholeExtent(0, 800 - 1, 0, 600 - 1, 0, 0);
	m_pImageImport->SetDataExtentToWholeExtent();
	m_pImageImport->SetDataScalarTypeToUnsignedChar();
	m_pImageImport->SetNumberOfScalarComponents(3);
	m_pImageImport->Update();

	// Create and configure background image actor
	m_pImageActor = vtkImageActor::New();
	m_pImageActor->SetInterpolate(1);
	m_pImageActor->SetInput(m_pImageImport->GetOutput());
	m_pImageActor->SetScale(1, -1, -1);
	m_pImageActor->VisibilityOn();

	// Add the image actor
	m_pSceneRenderer->AddActor(m_pImageActor);

	// Scale
	m_pSceneRenderer->GetActiveCamera()->SetParallelScale(600.0f);

	// Start the timer
	m_Timer.start(1000.0f / 60.0f);

	emit RenderBegin();
}

void CMainWindow::OnRenderEnd(void)
{
	qDebug("Rendering ended");

	m_LightingDockWidget.setEnabled(false);
	m_AppearanceDockWidget.setEnabled(false);
	m_StatisticsDockWidget.setEnabled(false);
	m_CameraDockWidget.setEnabled(false);
	m_SettingsDockWidget.setEnabled(false);

	m_pImageImport->Delete();

	m_pSceneRenderer->RemoveActor(m_pImageActor);

	m_pImageActor->Delete();

	m_pImageImport = NULL;
	m_pImageActor = NULL;

	// Stop the timer
	m_Timer.stop();

	emit RenderEnd();
}


