#pragma once

class QDemoWidget : public QWidget
{
	Q_OBJECT

public:
	QDemoWidget(const QString& Name, const QString& Description, const QString& Image, QWidget* pParent = NULL);
	virtual ~QDemoWidget(void);

	QGridLayout		m_MainLayout;
	QPushButton		m_Demo;
	QLabel			m_Name;
	QLabel			m_Description;
};

class QStartupDialog : public QDialog
{
	Q_OBJECT

public:
	QStartupDialog(QWidget* pParent = NULL);
	virtual ~QStartupDialog(void);

	virtual void accept();
	virtual QSize sizeHint() const;

public:

private:
	void LoadReadMe(const QString& FileName);

private slots:
	void OnLoadDemo(const QString& FileName);

signals:
	void LoadDemo(const QString& FileName);

private:
	QGridLayout			m_MainLayout;
	QGroupBox			m_DemoFilesGroupBox;
	QGridLayout			m_DemoFilesLayout;
	QLabel				m_ResampleNote;
	QGroupBox			m_ReadMeGroupBox;
	QGridLayout			m_ReadMeLayout;
	QTextEdit			m_ReadMe;
	QDialogButtonBox	m_DialogButtons;
	QCheckBox			m_ShowNextTime;
};