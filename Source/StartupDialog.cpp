
// Precompiled headers
#include "Stable.h"

#include "StartupDialog.h"

QStartupDialog::QStartupDialog(QWidget* pParent) :
	QDialog(pParent),
	m_MainLayout(),
	m_DemoFilesGroupBox(),
	m_DemoFilesLayout(),
	m_ResampleNote(),
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

	m_DemoFilesLayout.setAlignment(Qt::AlignTop);
	m_DemoFilesLayout.addWidget(new QDemoWidget("Manix", "Loads the bonsai data set, along with transfer function, lighting and camera presets<br><a href='http://www.osirix-viewer.com/Downloads.html'>Osirix Website</a>", ""), 0, 0);
	m_DemoFilesLayout.addWidget(new QDemoWidget("Backpack", "Loads the bonsai data set, along with transfer function, lighting and camera presets<br><a href='http://www.volvis.org/'>Volvis Website</a>", ""), 0, 1);
	m_DemoFilesLayout.addWidget(new QDemoWidget("Bonsai", "Loads the bonsai data set, along with transfer function, lighting and camera presets<br><a href='http://www9.informatik.uni-erlangen.de/External/vollib/'>Volume Library</a>", ""), 1, 0);
	m_DemoFilesLayout.addWidget(new QDemoWidget("Macoessix", "Loads the bonsai data set, along with transfer function, lighting and camera presets<br><a href='http://www.osirix-viewer.com/Downloads.html'>Osirix Website</a>", ""), 1, 1);
	m_DemoFilesLayout.addWidget(new QDemoWidget("Engine", "Loads the bonsai data set, along with transfer function, lighting and camera presets<br><a href='http://www.volvis.org/'>Volvis Website</a>", ""), 2, 0);
	m_DemoFilesLayout.addWidget(new QDemoWidget("Artifix", "Loads the bonsai data set, along with transfer function, lighting and camera presets<br><a href='http://www.osirix-viewer.com/Downloads.html'>Osirix Website</a>", ""), 2, 1);

	m_ResampleNote.setText("In order to reduce the size of the installer we distribute resampled volumes (sampled at 50%). The original volumes, as well as other volumes can be downloaded from the <a href='http://code.google.com/p/exposure-render/downloads'>Exposure Render Website</a>");
	m_ResampleNote.setWordWrap(true);
	m_ResampleNote.setContentsMargins(10, 10, 10, 2);
	
	m_DemoFilesLayout.addWidget(&m_ResampleNote, 3, 0, 1, 2);

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
// 	pSignalMapper->setMapping(&m_Demo1, QString("Bonsai.mhd"));
// 	pSignalMapper->setMapping(&m_Demo2, QString("Manix.mhd"));
// 	pSignalMapper->setMapping(&m_Demo3, QString("Backpack.mhd"));
// 	pSignalMapper->setMapping(&m_Demo4, QString("Backpack.mhd"));

// 	connect(&m_Demo1, SIGNAL(clicked()), pSignalMapper, SLOT (map()));
// 	connect(&m_Demo2, SIGNAL(clicked()), pSignalMapper, SLOT (map()));
// 	connect(&m_Demo3, SIGNAL(clicked()), pSignalMapper, SLOT (map()));
// 	connect(&m_Demo4, SIGNAL(clicked()), pSignalMapper, SLOT (map()));

	connect(pSignalMapper, SIGNAL(mapped(const QString&)), this, SLOT(OnLoadDemo(const QString&)));
};

QStartupDialog::~QStartupDialog(void)
{
	QSettings Settings;

	Settings.setValue("startup/dialog/show", m_ShowNextTime.isChecked());
}

QSize QStartupDialog::sizeHint() const
{
	return QSize(600, 300);
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

QDemoWidget::QDemoWidget(const QString& Name, const QString& Description, const QString& Image, QWidget* pParent /*= NULL*/) :
	QWidget(pParent),
	m_MainLayout(),
	m_Demo(),
	m_Name(),
	m_Description()
{
	setLayout(&m_MainLayout);

	m_Demo.setFixedSize(72, 72);

	m_Name.setWordWrap(true);
	m_Name.setText("<b>" + Name + "</b>");

	m_Description.setWordWrap(true);
	m_Description.setText(Description);

	m_MainLayout.addWidget(&m_Demo, 0, 0, 2, 1, Qt::AlignTop);
	m_MainLayout.addWidget(&m_Name, 0, 1, Qt::AlignTop);
	m_MainLayout.addWidget(&m_Description, 1, 1, Qt::AlignTop);
}

QDemoWidget::~QDemoWidget(void)
{
}
