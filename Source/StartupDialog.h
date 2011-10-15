#pragma once

#include "HardwareInfoWidget.h"

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