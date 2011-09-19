#pragma once

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
	QPushButton			m_Demo1;
	QLabel				m_Label1;
	QPushButton			m_Demo2;
	QLabel				m_Label2;
	QPushButton			m_Demo3;
	QLabel				m_Label3;
	QPushButton			m_Demo4;
	QLabel				m_Label4;
	QGroupBox			m_ReadMeGroupBox;
	QGridLayout			m_ReadMeLayout;
	QTextEdit			m_ReadMe;
	QDialogButtonBox	m_DialogButtons;
	QCheckBox			m_ShowNextTime;
};