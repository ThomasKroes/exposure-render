/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
	void DownloadVersionInfo(void);

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