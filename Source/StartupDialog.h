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

#include "HardwareWidget.h"

class QStartupDialog : public QDialog
{
	Q_OBJECT

public:
	QStartupDialog(QWidget* pParent = NULL);
	virtual ~QStartupDialog(void);

	virtual void accept();
	virtual QSize sizeHint() const;

public:
	void LoadDemoFile(const QString& BaseName);

private:
	void LoadReadMe(const QString& FileName);

signals:
	void LoadDemo(const QString& FileName);

private:
	QGridLayout			m_MainLayout;
	QGroupBox			m_DemoFilesGroupBox;
	QGridLayout			m_DemoFilesLayout;
	QLabel				m_ResampleNote;
	QHardwareWidget		m_HardwareWidget;
	QGroupBox			m_ReadMeGroupBox;
	QGridLayout			m_ReadMeLayout;
	QTextEdit			m_ReadMe;
	QDialogButtonBox	m_DialogButtons;
	QCheckBox			m_ShowNextTime;
};

class QDemoWidget : public QWidget
{
	Q_OBJECT

public:
	QDemoWidget(QStartupDialog* pStartupDialog, const QString& NameUI, const QString& BaseName, const QString& Description, const QString& Image, QWidget* pParent = NULL);
	virtual ~QDemoWidget(void);

private:
	QStartupDialog*		m_pStartupDialog;
	QGridLayout			m_MainLayout;
	QPushButton			m_Demo;
	QLabel				m_Name;
	QLabel				m_Description;
	QString				m_BaseName;

private slots:
	void OnLoadDemo(void);
};