
#include "PresetsWidget.h"

QPresetsWidget::QPresetsWidget(const QString& PresetFileName, QWidget* pParent /*= NULL*/ ) :
	QGroupBox(pParent),
	m_PresetFileName(PresetFileName),
	m_MainLayout(),
	m_PresetName(),
	m_LoadPreset(),
	m_SavePreset(),
	m_RemovePreset(),
	m_LoadPresets(),
	m_SavePresets(),
	m_Presets()
{
	// Title, status and tooltip
	setTitle("Presets");
	setToolTip("Presets");
	setStatusTip("Presets");

	m_PresetFileName = PresetFileName;

	CreateUI();
	CreateConnections();

	// Load transfer function presets from file
	LoadPresetsFromFile();
}

QPresetsWidget::~QPresetsWidget(void)
{
	// Save transfer function presets to file
	SavePresetsToFile();
}

void QPresetsWidget::CreateUI(void)
{
	// Assign layout
	setLayout(&m_MainLayout);

	// Name edit
	m_PresetName.setEditable(true);
	m_MainLayout.addWidget(&m_PresetName, 0, 0);

	// Load preset
	m_LoadPreset.setText("L");
	m_LoadPreset.setToolTip("Load preset");
	m_LoadPreset.setStatusTip("Load transfer function preset");
	m_LoadPreset.setFixedWidth(20);
	m_LoadPreset.setFixedHeight(20);
	m_MainLayout.addWidget(&m_LoadPreset, 0, 1);

	// Save Preset
	m_SavePreset.setEnabled(false);
	m_SavePreset.setText("S");
	m_SavePreset.setToolTip("Save Preset");
	m_SavePreset.setStatusTip("Save transfer function preset");
	m_SavePreset.setFixedWidth(20);
	m_SavePreset.setFixedHeight(20);
	m_MainLayout.addWidget(&m_SavePreset, 0, 2);

	// Remove preset
	m_RemovePreset.setEnabled(false);
	m_RemovePreset.setText("R");
	m_RemovePreset.setToolTip("Remove Preset");
	m_RemovePreset.setStatusTip("Remove transfer function preset");
	m_RemovePreset.setFixedWidth(20);
	m_RemovePreset.setFixedHeight(20);
	m_MainLayout.addWidget(&m_RemovePreset, 0, 3);

	// Load presets
	m_LoadPresets.setText("LF");
	m_LoadPresets.setToolTip("Load presets from files");
	m_LoadPresets.setStatusTip("Load transfer function presets from file");
	m_LoadPresets.setFixedWidth(20);
	m_LoadPresets.setFixedHeight(20);
	m_MainLayout.addWidget(&m_LoadPresets, 0, 4);

	// Save presets
	m_SavePresets.setText("SF");
	m_SavePresets.setToolTip("Save presets to file");
	m_SavePresets.setStatusTip("Save transfer function presets to file");
	m_SavePresets.setFixedWidth(20);
	m_SavePresets.setFixedHeight(20);
	m_MainLayout.addWidget(&m_SavePresets, 0, 5);
}

void QPresetsWidget::CreateConnections(void)
{
	connect(&m_LoadPreset, SIGNAL(clicked()), this, SLOT(OnLoadPreset()));
	connect(&m_SavePreset, SIGNAL(clicked()), this, SLOT(OnSavePreset()));
	connect(&m_PresetName, SIGNAL(returnPressed()), this, SLOT(OnSavePreset()));
	connect(&m_RemovePreset, SIGNAL(clicked()), this, SLOT(OnRemovePreset()));
	connect(&m_LoadPresets, SIGNAL(clicked()), this, SLOT(OnLoadPresets()));
	connect(&m_SavePresets, SIGNAL(clicked()), this, SLOT(OnSavePresets()));
	connect(&m_PresetName, SIGNAL(editTextChanged(const QString&)), this, SLOT(OnPresetNameChanged(const QString&)));
}

void QPresetsWidget::LoadPresetsFromFile(const bool& ChoosePath)
{
	qDebug("Loading transfer function presets from file");

	// XML file containing transfer function presets
	QFile XmlFile;

	// Get applications working directory
	QString CurrentPath = QDir::currentPath();

	// Set the file name
	if (ChoosePath)
	{
		// Create open file dialog
		XmlFile.setFileName(QFileDialog::getOpenFileName(this, "Load" + m_PresetFileName + "from file", "", tr("XML Files (*.xml)")));
	}
	else
	{
		XmlFile.setFileName(CurrentPath + "/" + m_PresetFileName);
	}
	
	// Open the XML file
	if (!XmlFile.open(QIODevice::ReadOnly))
	{
		qDebug("Failed to open file for reading.");
		return;
	}

	// Document object model for XML
	QDomDocument DOM(m_PresetFileName);

	// Parse file content into DOM
	if (!DOM.setContent(&XmlFile))
	{
		qDebug("Failed to parse file into a DOM tree.");
		XmlFile.close();
    	return;
	}
 
	// Obtain document root node
	QDomElement DomRoot = DOM.documentElement();

	LoadPresets(DomRoot);

	XmlFile.close();

	// Update the presets list
	UpdatePresetsList();
}

void QPresetsWidget::SavePresetsToFile(const bool& ChoosePath)
{
	qDebug("Saving transfer function presets to file");

	// XML file containing transfer function presets
	QFile XmlFile;

	// Get applications working directory
	QString CurrentPath = QDir::currentPath();

	// Set the file name
	if (ChoosePath)
	{
		// Create open file dialog
		XmlFile.setFileName(QFileDialog::getSaveFileName(this, "Save" + m_PresetFileName + "to file", "", tr("XML Files (*.xml)")));
	}
	else
	{
		XmlFile.setFileName(CurrentPath + "/" + m_PresetFileName);
	}
	
	// Open the XML file
	if (!XmlFile.open(QIODevice::WriteOnly ))
	{
		qDebug("Failed to open file for writing.");
		return;
	}

	// Document object model for XML
	QDomDocument DOM(m_PresetFileName);

	// Write each transfer function to the file
	QDomElement DomRoot = DOM.documentElement();

	SavePresets(DOM, DomRoot);

	// Create text stream
	QTextStream TextStream(&XmlFile);

	// Save the XML file
	DOM.save(TextStream, 0);

	// Close the XML file
	XmlFile.close();
}

void QPresetsWidget::LoadPresets(QDomElement& Root)
{
}

void QPresetsWidget::SavePresets(QDomDocument& DomDoc, QDomElement& Root)
{
}

void QPresetsWidget::LoadPreset(QPresetXML* pPreset)
{
}

void QPresetsWidget::SavePreset(const QString& Name)
{
}

void QPresetsWidget::UpdatePresetsList(void)
{
	for (int i = 0; i < m_Presets.size(); i++)
	{
		// Put pointer to preset in void pointer
		QVariant Variant = qVariantFromValue((void*)m_Presets[i]);

		m_PresetName.addItem(m_Presets[i]->GetName(), Variant);
	}
}

void QPresetsWidget::OnLoadPreset(void)
{
	if (m_PresetName.currentIndex() < 0)
		return;

	QVariant Variant = m_PresetName.itemData(m_PresetName.currentIndex());

	LoadPreset((QPresetXML*)Variant.value<void*>());
}

void QPresetsWidget::OnSavePreset(void)
{
	SavePreset(m_PresetName.lineEdit()->text());
}

void QPresetsWidget::OnRemovePreset(void)
{
	/*
	// Get current row
	const int CurrentRow = m_PresetList.currentRow();

	if (CurrentRow < 0)
		return;

	m_TransferFunctions.removeAt(CurrentRow);

	UpdatePresetsList();
	*/
}

void QPresetsWidget::OnLoadPresets(void)
{
	// Clear the current list of presets
//	m_TransferFunctions.clear();

	// Load the presets, asking for a location
	LoadPresetsFromFile(true);
}

void QPresetsWidget::OnSavePresets(void)
{
	// Save the presets, asking for a location
	SavePresetsToFile(true);
}

void QPresetsWidget::OnDummy(void)
{
	// Save the presets, asking for a location
	SavePresetsToFile(true);
}

void QPresetsWidget::OnPresetNameChanged(const QString& Text)
{
	m_SavePreset.setEnabled(Text.length() > 0);
}

void QPresetsWidget::OnPresetItemChanged(QListWidgetItem* pWidgetItem)
{
	QPresetItem* pPresetItem = dynamic_cast<QPresetItem*>(pWidgetItem);
	//(YourClass *) v.value<void *>();
//	if (pPresetItem)
//		((QTransferFunction*)pPresetItem->m_pData)->SetName(pWidgetItem->text());
}




