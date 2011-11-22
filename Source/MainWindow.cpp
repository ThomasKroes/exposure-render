/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "MainWindow.h"

#include "VtkWidget.h"
#include "StartupDialog.h"
#include "AboutDialog.h"

QUrl	gVersionInfoUrl("http://exposure-render.googlecode.com/hg/VersionInfo.xml");
int		gVersionID = 303;

// Main window singleton
CMainWindow* gpMainWindow = NULL;

CMainWindow::CMainWindow() :
	QMainWindow(),
	m_CurrentFile(""),
    m_pFileMenu(NULL),
    m_pHelpMenu(NULL),
    m_pFileToolBar(NULL),
	m_pPlaybackToolBar(),
	m_LogDockWidget(),
	m_LightingDockWidget(),
	m_AppearanceDockWidget(),
	m_StatisticsDockWidget(),
	m_CameraDockWidget(),
	m_SettingsDockWidget(),
	m_SlicingDockWidget(),
	m_RenderOutputDialog()
{
	gpMainWindow = this;

	setCentralWidget(&m_RenderView);

//	m_RenderView.m_RenderWidget.SetActive();
	m_RenderView.SetActive();
//	m_RenderOutputDialog.m_VtkRenderWidget.SetSource(&m_RenderWidget);

	CreateMenus();
	CreateStatusBar();
	SetupDockingWidgets();

    setUnifiedTitleAndToolBarOnMac(true);

	setWindowState(Qt::WindowMaximized);

	setWindowFilePath(QString());

	m_Update.setOpenExternalLinks(true);

	QObject::connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	QObject::connect(&gStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	QObject::connect(&m_HttpGet, SIGNAL(done()), this, SLOT(VersionInfoDownloaded()));

	OnCheckForUpdates();
}

CMainWindow::~CMainWindow(void)
{
//	KillRenderThread();
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

	m_pFileMenu->addAction("Close", this, SLOT(Close()));

	m_pFileMenu->addSeparator();

	m_pFileMenu->addAction(GetIcon("image-export"), "Save Image", this, SLOT(OnSaveImage()));

    m_pFileMenu->addSeparator();

	m_pFileMenu->addAction(GetIcon("star"), "Welcome screen", this, SLOT(ShowStartupDialog()));

	m_pFileMenu->addSeparator();

    m_pFileMenu->addAction(GetIcon("door--arrow"), "Exit", this, SLOT(close()));
    
	menuBar()->addSeparator();

	m_pViewMenu = menuBar()->addMenu(tr("&View"));
	
    menuBar()->addSeparator();

    m_pHelpMenu = menuBar()->addMenu(tr("&Help"));
    m_pHelpMenu->addAction(GetIcon("question"), "About Exposure Render", this, SLOT(About()));
	m_pHelpMenu->addAction(GetIcon("question-white"), "About Qt", qApp, SLOT(aboutQt()));

	m_pHelpMenu->addSeparator();

	m_pHelpMenu->addAction(GetIcon("globe"), "Visit Website", this, SLOT(OnVisitWebsite()));
//	m_pHelpMenu->addAction(GetIcon("globe"), "Check for updates", this, SLOT(OnCheckForUpdates()));

	UpdateRecentFileActions();
}

void CMainWindow::CreateStatusBar()
{
    statusBar()->showMessage(tr("Ready"));
	statusBar()->setSizeGripEnabled(true);
	statusBar()->addPermanentWidget(&m_Update);
	statusBar()->setStyleSheet("QStatusBar::item { border: 0px solid black };");
}

void CMainWindow::SetupDockingWidgets()
{
	m_LightingDockWidget.setEnabled(false);
    m_LightingDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::RightDockWidgetArea, &m_LightingDockWidget);
    m_pViewMenu->addAction(m_LightingDockWidget.toggleViewAction());

	m_AppearanceDockWidget.setEnabled(false);
	m_AppearanceDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
	addDockWidget(Qt::LeftDockWidgetArea, &m_AppearanceDockWidget);
    m_pViewMenu->addAction(m_AppearanceDockWidget.toggleViewAction());

	m_CameraDockWidget.setEnabled(false);
	m_CameraDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::RightDockWidgetArea, &m_CameraDockWidget);
    m_pViewMenu->addAction(m_CameraDockWidget.toggleViewAction());

	m_LogDockWidget.setEnabled(false);
	m_LogDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
	addDockWidget(Qt::RightDockWidgetArea, &m_LogDockWidget);
	m_pViewMenu->addAction(m_LogDockWidget.toggleViewAction());

	m_StatisticsDockWidget.setEnabled(false);
	m_StatisticsDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::RightDockWidgetArea, &m_StatisticsDockWidget);
    m_pViewMenu->addAction(m_StatisticsDockWidget.toggleViewAction());

	m_SlicingDockWidget.setEnabled(false);
	m_SlicingDockWidget.setAllowedAreas(Qt::AllDockWidgetAreas);
    addDockWidget(Qt::RightDockWidgetArea, &m_SlicingDockWidget);
    m_pViewMenu->addAction(m_SlicingDockWidget.toggleViewAction());
	
	tabifyDockWidget(&m_AppearanceDockWidget, &m_LightingDockWidget);

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
    QString FileName = GetOpenFileName2("Open volume", "Meta Image Volume Files (*.mhd)", "grid");

	// Exit empty
	if (FileName.isEmpty())
		return;

	// Open the file
	Open(FileName);
}

void CMainWindow::Open(QString FilePath)
{
	// Kill current rendering thread
//	KillRenderThread();

	// Window name update
	SetCurrentFile(FilePath);

	// Make string suitable for VTK
	FilePath.replace("/", "\\\\");

	gpActiveRenderWidget->LoadVolume(FilePath);

	gStatus.SetRenderBegin();
}

void CMainWindow::OnLoadDemo(const QString& FileName)
{
	Open(QApplication::applicationDirPath() + "/Examples/" + FileName);
}

void CMainWindow::Close(void)
{
//	KillRenderThread();

//	Log("Device memory: " + QString::number(GetUsedCudaMemory() / MB, 'f', 2) + "/" + QString::number(GetTotalCudaMemory() / MB, 'f', 2) + " MB", "memory");

	m_RenderOutputDialog.close();
}

void CMainWindow::Exit(void)
{
	Close();
}

void CMainWindow::About()
{
	QAboutDialog AboutDialog;
	AboutDialog.exec();
}

void CMainWindow::OnRenderBegin(void)
{
	Log("Rendering started", "control");

	m_LightingDockWidget.setEnabled(true);
	m_AppearanceDockWidget.setEnabled(true);
	m_StatisticsDockWidget.setEnabled(true);
	m_CameraDockWidget.setEnabled(true);
	m_SettingsDockWidget.setEnabled(true);
	m_LogDockWidget.setEnabled(true);
	m_SlicingDockWidget.setEnabled(true);

//	m_RenderOutputDialog.show();
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
	m_SlicingDockWidget.setEnabled(false);

//	m_RenderOutputDialog.hide();
//	m_RenderOutputDialog.close();
}

void CMainWindow::ShowStartupDialog(void)
{
	Close();

	QStartupDialog StartupDialog;

	connect(&StartupDialog, SIGNAL(LoadDemo(const QString&)), gpMainWindow, SLOT(OnLoadDemo(const QString&)));

	StartupDialog.exec();
}

void CMainWindow::OnVisitWebsite(void)
{
	QDesktopServices::openUrl(QUrl("http://code.google.com/p/exposure-render/"));
}

void CMainWindow::OnSaveImage(void)
{
	/*
	if (!gpRenderThread)
		return;

	gpRenderThread->PauseRendering(true);

	SaveImage((unsigned char*)gpRenderThread->m_pRenderImage, gScene.m_Camera.m_Film.m_Resolution.GetResX(), gScene.m_Camera.m_Film.m_Resolution.GetResY());

	gpRenderThread->PauseRendering(false);
	*/
}

void CMainWindow::OnCheckForUpdates(void)
{
	const QString FilePath = QApplication::applicationDirPath() + "/" + QFileInfo(gVersionInfoUrl.path()).fileName();

	m_HttpGet.GetFile(gVersionInfoUrl, FilePath);
}

void CMainWindow::VersionInfoDownloaded(void)
{
	Log("Version info downloaded", "globe");

	const QString FilePath = QApplication::applicationDirPath() + "/" + QFileInfo(gVersionInfoUrl.path()).fileName();

	QFile XmlFile;

	XmlFile.setFileName(FilePath);

	if (!XmlFile.open(QIODevice::ReadOnly))
	{
		Log(QString("Failed to open " + QFileInfo(FilePath).fileName() + " for reading: " + XmlFile.errorString()).toAscii(), QLogger::Critical);
		return;
	}

	QDomDocument DOM;

	if (!DOM.setContent(&XmlFile))
	{
		Log("Failed to parse " + QFileInfo(FilePath).fileName() + "into a DOM tree.", QLogger::Critical);
		XmlFile.close();
		return;
	}

	QDomElement Root = DOM.documentElement();

	QDomNodeList Versions = Root.elementsByTagName("Version");

	bool FoundNewerVersion = false;

	QString FileUrl, Name;

	for (int i = 0; i < Versions.count(); i++)
	{
		QDomNode Version = Versions.item(i);

		const int ID = Version.firstChildElement("ID").text().toInt();

		if (ID > gVersionID)
		{
			FoundNewerVersion = true;
			
			FileUrl = Version.firstChildElement("FileUrl").text();
			Name	= Version.firstChildElement("Name").text();
		}
	}

	XmlFile.close();

	if (FoundNewerVersion)
	{
		Log("Found a new version of Exposure Render", "globe");

		m_Update.setText("There is a new version of Exposure Render available, click <a href='" + FileUrl + "'>here</a> to download it");
	}
}