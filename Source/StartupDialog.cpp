
// Precompiled headers
#include "Stable.h"

#include "StartupDialog.h"

QStartupDialog::QStartupDialog(QWidget* pParent) :
	QDialog(pParent),
	m_MainLayout(),
	m_Demo1(),
	m_Label1(),
	m_Demo2(),
	m_Label2(),
	m_Demo3(),
	m_Label3(),
	m_ReadMe(),
	m_DialogButtons(),
	m_ShowNextTime()
{
	resize(400, 100);

	setWindowTitle("Welcome");
	setWindowIcon(GetIcon("star"));

	setLayout(&m_MainLayout);

	m_Demo1.setText("Load Demo 1");
	m_Demo2.setText("Load Demo 2");
	m_Demo3.setText("Load Demo 3");

	m_Label1.setText("Demonstration file 1");
	m_Label2.setText("Demonstration file 2");
	m_Label3.setText("Demonstration file 3");

	m_MainLayout.addWidget(&m_Demo1, 0, 0);
	m_MainLayout.addWidget(&m_Label1, 0, 1);
	m_MainLayout.addWidget(&m_Demo2, 1, 0);
	m_MainLayout.addWidget(&m_Label2, 1, 1);
	m_MainLayout.addWidget(&m_Demo3, 2, 0);
	m_MainLayout.addWidget(&m_Label3, 2, 1);

	// Add read me text
	m_MainLayout.addWidget(&m_ReadMe, 3, 0, 1, 2);

	// Standard dialog buttons
	m_DialogButtons.setStandardButtons(QDialogButtonBox::Ok);

	// Show next time button
	m_ShowNextTime.setText(tr("Show this dialog at start up"));

	// Checked
	m_ShowNextTime.setChecked(true);

	m_MainLayout.addWidget(&m_ShowNextTime, 4, 0);

	// Load the read me fiel
	LoadReadMe("Readme.txt");
};

QStartupDialog::~QStartupDialog(void)
{
	QSettings Settings;

	Settings.setValue("startup/dialog/show", m_ShowNextTime.isChecked());
}

QSize QStartupDialog::sizeHint() const
{
	return QSize(450, 450);
}

void QStartupDialog::accept()
{
	QSettings Settings;

	QDialog::accept();
}

void QStartupDialog::LoadDemo(void)
{

}

void QStartupDialog::LoadReadMe(const QString& FileName)
{
	QFile File(QApplication::applicationDirPath() + "/Readme.txt");

	if (!File.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		Log("Unable to open " + File.fileName(), QLogger::Critical);
		return;
	}

	QByteArray DocumentArray;
	DocumentArray = File.readAll();

	m_ReadMe.setPlainText(DocumentArray);
}




