
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
	setWindowTitle("Welcome");
	setWindowIcon(GetIcon("star"));

	setLayout(&m_MainLayout);

	m_Demo1.setText("Load Demo 1");
	m_Demo2.setText("Load Demo 2");
	m_Demo3.setText("Load Demo 3");

	m_Label1.setText("Loads the bonsai data set, along with transfer function, lighting and camera presets");
	m_Label2.setText("Loads the manix data set, along with transfer function, lighting and camera presets");
	m_Label3.setText("Loads the backpack data set, along with transfer function, lighting and camera presets");

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

	connect(&m_DialogButtons, SIGNAL(accepted()), this, SLOT(accept()));

	// Show next time button
	m_ShowNextTime.setText(tr("Show this dialog at start up"));

	// Checked
	m_ShowNextTime.setChecked(true);

	m_MainLayout.addWidget(&m_ShowNextTime, 4, 0);
	m_MainLayout.addWidget(&m_DialogButtons, 4, 1);

	// Load the read me file
	LoadReadMe("Readme.txt");

	QSignalMapper* pSignalMapper = new QSignalMapper(this);
	pSignalMapper->setMapping(&m_Demo1, QString("Bonsai.mhd"));
	pSignalMapper->setMapping(&m_Demo2, QString("manix.mhd"));
	pSignalMapper->setMapping(&m_Demo3, QString("backpack.mhd"));

	connect(&m_Demo1, SIGNAL(clicked()), pSignalMapper, SLOT (map()));
	connect(&m_Demo2, SIGNAL(clicked()), pSignalMapper, SLOT (map()));
	connect(&m_Demo3, SIGNAL(clicked()), pSignalMapper, SLOT (map()));

	connect(pSignalMapper, SIGNAL(mapped(const QString&)), this, SLOT(OnLoadDemo(const QString&)));
};

QStartupDialog::~QStartupDialog(void)
{
	QSettings Settings;

	Settings.setValue("startup/dialog/show", m_ShowNextTime.isChecked());
}

QSize QStartupDialog::sizeHint() const
{
	return QSize(250, 450);
}

void QStartupDialog::accept()
{
	QDialog::accept();
}

void QStartupDialog::OnLoadDemo(const QString& FileName)
{
	emit LoadDemo(FileName);
	accept();
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