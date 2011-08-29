
#include <QtGui>

#include "MainWindow.h"
#include "TransferFunctionWidget.h"
#include "LightingDockWidget.h"
#include "VolumeAppearanceDockWidget.h"
#include "StatisticsDockWidget.h"
#include "CameraDockWidget.h"
#include "SettingsDockWidget.h"
#include "VtkWidget.h"
#include "LoadVolume.h"

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
	m_pLightingDockWidget(NULL),
	m_pVolumeAppearanceDockWidget(NULL),
	m_pStatisticsDockWidget(NULL),
	m_pCameraDockWidget(NULL),
	m_pSettingsDockWidget(NULL),
    m_pOpenAct(NULL),
	m_pCloseAct(NULL),
    m_pExitAct(NULL),
    m_pAboutAct(NULL),
    m_pAboutQtAct(NULL),
	m_pSeparatorAction(NULL),
	m_Timer()
{
	// Create VTK rendering window
	m_pVtkWidget = new CVtkWidget();
	setCentralWidget(m_pVtkWidget);

	CreateActions();
	CreateMenus();
	CreateToolBars();
	CreateStatusBar();
	CreateDockWindows();

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
}

void CMainWindow::CreateDockWindows()
{
	// Lighting dock widget
    m_pLightingDockWidget = new CLightingDockWidget(this);
    m_pLightingDockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::RightDockWidgetArea, m_pLightingDockWidget);
    m_pViewMenu->addAction(m_pLightingDockWidget->toggleViewAction());

	// Appearance dock widget
    m_pVolumeAppearanceDockWidget = new CVolumeAppearanceDockWidget(this);
	m_pVolumeAppearanceDockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);
	addDockWidget(Qt::LeftDockWidgetArea, m_pVolumeAppearanceDockWidget);
    m_pViewMenu->addAction(m_pVolumeAppearanceDockWidget->toggleViewAction());

	// Statistics dock widget
    m_pStatisticsDockWidget = new CStatisticsDockWidget(this);
	m_pStatisticsDockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::LeftDockWidgetArea, m_pStatisticsDockWidget);
    m_pViewMenu->addAction(m_pStatisticsDockWidget->toggleViewAction());

	// Camera dock widget
    m_pCameraDockWidget = new CCameraDockWidget(this);
	m_pCameraDockWidget->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    addDockWidget(Qt::RightDockWidgetArea, m_pCameraDockWidget);
    m_pViewMenu->addAction(m_pCameraDockWidget->toggleViewAction());

	// Settings dock widget
    m_pSettingsDockWidget = new CSettingsDockWidget(this);
	m_pSettingsDockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::RightDockWidgetArea, m_pSettingsDockWidget);
    m_pViewMenu->addAction(m_pSettingsDockWidget->toggleViewAction());
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
    QString FileName = QFileDialog::getOpenFileName(this, tr("Open volume"), "", tr("Volume Files (*.mhd)"));

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

	// Create new render thread and run it
	gpRenderThread = new CRenderThread(FileName, this);
	

	// Load the VTK volume
	if (LoadVtkVolume(FileName.toAscii().data(), gpScene, gpRenderThread->m_pImageDataVolume))
	{
		gpRenderThread->m_Loaded = true;
	}
	else
	{
		return;
	}

	// Force the render thread to allocate the necessary buffers, do not remove this line
	gpScene->m_DirtyFlags.SetFlag(FilmResolutionDirty | CameraDirty);

	gThreadAlive = true;
	gpRenderThread->start();
//	gpRenderThread->setPriority(QThread::Priority::LowestPriority);

	
}

void CMainWindow::SetupRenderView(void)
{
	// Create and configure image importer
	m_pImageImport = vtkImageImport::New();
	m_pImageImport->SetDataSpacing(1, 1, 1);
	m_pImageImport->SetDataOrigin(0, 0, 0);
	m_pImageImport->SetImportVoidPointer((void*)malloc(3 * 512 * 512 * sizeof(unsigned char)));//gpRenderThread->RenderImage());
	m_pImageImport->SetWholeExtent(0, 512 - 1, 0, 512 - 1, 0, 0);
	m_pImageImport->SetDataExtentToWholeExtent();
	m_pImageImport->SetDataScalarTypeToUnsignedChar();
	m_pImageImport->SetNumberOfScalarComponents(3);
	m_pImageImport->Update();

	// Create and configure background image actor
	m_pImageActor = vtkImageActor::New();
	m_pImageActor->SetInterpolate(1);
	m_pImageActor->SetInput(m_pImageImport->GetOutput());
	m_pImageActor->SetScale(2, -2, -2);
	m_pImageActor->VisibilityOn();

	// Create and configure scene renderer
	m_pSceneRenderer = vtkRenderer::New();
	m_pSceneRenderer->SetBackground(0.4, 0.4, 0.43);
	m_pSceneRenderer->SetBackground2(0.6, 0.6, 0.6);
	m_pSceneRenderer->SetGradientBackground(true);
//	m_pSceneRenderer->GetActiveCamera()->SetPosition(1.0, 1.0, 1.0);
//	m_pSceneRenderer->GetActiveCamera()->SetFocalPoint(0.0, 0.0, 0.0);
//	m_pSceneRenderer->GetActiveCamera()->SetRoll(0.0);
	m_pSceneRenderer->AddActor(m_pImageActor);

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

	// Create a timer
//	m_pRenderWindow->GetInteractor()->CreateRepeatingTimer(10.0f);

	m_Timer.start(1000.0f / 30.0f);
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
}

void CMainWindow::Exit()
{
	
}

void CMainWindow::About()
{
	QColorDialog Dia(this);

	Dia.exec();

 //  QMessageBox::about(this, tr("About Exposure Render"),
 //           tr("This application illustrates the concepts from the paper: <b>Raytraced lighting in direct volume rendering</b>\n"
	//		"For more information visit: <b>graphics.tudelft.nl</b>"));
}