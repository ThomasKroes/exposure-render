
// Precompiled headers
#include "Stable.h"

#include "StartupDialog.h"

QStartupDialog::QStartupDialog(QWidget* pParent) :
	QDialog(pParent),
	m_MainLayout(),
	m_DemoFilesGroupBox(),
	m_DemoFilesLayout(),
	m_Demo1(),
	m_Label1(),
	m_Demo2(),
	m_Label2(),
	m_Demo3(),
	m_Label3(),
	m_ReadMeGroupBox(),
	m_ReadMeLayout(),
	m_ReadMe(),
	m_DialogButtons(),
	m_ShowNextTime()
{
	setWindowTitle("Welcome");
	setWindowIcon(GetIcon("star"));

	setLayout(&m_MainLayout);

	m_MainLayout.addWidget(&m_DemoFilesGroupBox, 0, 0, 1, 2);

	m_DemoFilesGroupBox.setLayout(&m_DemoFilesLayout);
	m_DemoFilesGroupBox.setTitle("Demo Files");
	m_DemoFilesGroupBox.setToolTip("Demo Files");
	m_DemoFilesGroupBox.setStatusTip("Demo files");

	m_Demo1.setFixedSize(50, 50);
	m_Demo2.setFixedSize(50, 50);
	m_Demo3.setFixedSize(50, 50);

	m_Demo1.setText("");
	m_Demo2.setText("");
	m_Demo3.setText("");

	m_Demo1.setToolTip("Bonsai");
	m_Demo2.setToolTip("Manix");
	m_Demo3.setToolTip("Backpack");

	m_Label1.setText("<b>Bonsai</b><br>Loads the bonsai data set, along with transfer function, lighting and camera presets");
	m_Label2.setText("<b>Manix</b><br>Loads the manix data set, along with transfer function, lighting and camera presets");
	m_Label3.setText("<b>Backpack</b><br>Loads the backpack data set, along with transfer function, lighting and camera presets");

	m_DemoFilesLayout.addWidget(&m_Demo1, 0, 0);
	m_DemoFilesLayout.addWidget(&m_Label1, 0, 1);
	m_DemoFilesLayout.addWidget(&m_Demo2, 1, 0);
	m_DemoFilesLayout.addWidget(&m_Label2, 1, 1);
	m_DemoFilesLayout.addWidget(&m_Demo3, 2, 0);
	m_DemoFilesLayout.addWidget(&m_Label3, 2, 1);

	m_MainLayout.addWidget(&m_ReadMeGroupBox, 1, 0, 1, 2);

	m_ReadMeGroupBox.setLayout(&m_ReadMeLayout);
	m_ReadMeGroupBox.setTitle("Special Notes");
	m_ReadMeGroupBox.setToolTip("Special Notes");
	m_ReadMeGroupBox.setStatusTip("Special Notes");

	m_ReadMe.setEnabled(false);

	// Add read me text
	m_ReadMeLayout.addWidget(&m_ReadMe, 0, 0);

	// Standard dialog buttons
	m_DialogButtons.setStandardButtons(QDialogButtonBox::Ok);

	connect(&m_DialogButtons, SIGNAL(accepted()), this, SLOT(accept()));

	// Show next time button
	m_ShowNextTime.setText(tr("Show this dialog at start up"));

	// Checked
	m_ShowNextTime.setChecked(true);

	m_MainLayout.addWidget(&m_ShowNextTime, 2, 0);
	m_MainLayout.addWidget(&m_DialogButtons, 2, 1);

	// Load the read me file
	LoadReadMe("Readme.txt");

	QSignalMapper* pSignalMapper = new QSignalMapper(this);
	pSignalMapper->setMapping(&m_Demo1, QString("Bonsai.mhd"));
	pSignalMapper->setMapping(&m_Demo2, QString("Manix.mhd"));
	pSignalMapper->setMapping(&m_Demo3, QString("Backpack.mhd"));

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