
#include "TransferFunctionPresetsWidget.h"

QTransferFunctionPresetsWidget::QTransferFunctionPresetsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_PresetName(),
	m_LoadPreset(),
	m_SavePreset(),
	m_RemovePreset(),
	m_LoadPresets(),
	m_SavePresets(),
	m_Dummy(),
	m_PresetList(),
	m_TransferFunctions()
{
	// Title, status and tooltip
	setTitle("Presets");
	setToolTip("Transfer Function Presets");
	setStatusTip("Transfer Function Presets");

	CreateUI();
	CreateConnections();

	// Load transfer function presets from file
	LoadPresetsFromFile();
}

QTransferFunctionPresetsWidget::~QTransferFunctionPresetsWidget(void)
{
	// Save transfer function presets to file
	SavePresetsToFile();
}

void QTransferFunctionPresetsWidget::CreateUI(void)
{
	// Create grid layout
	m_GridLayout.setAlignment(Qt::AlignTop);

	// Assign layout
	setLayout(&m_GridLayout);

	// Presets list
	m_PresetList.setParent(this);
	m_PresetList.setSelectionMode(QAbstractItemView::SingleSelection);
	m_PresetList.setAlternatingRowColors(true);
	m_PresetList.setSortingEnabled(true);

	m_GridLayout.addWidget(&m_PresetList, 1, 0, 6, 1);

	// Name edit
	m_PresetName.setParent(this);
	m_GridLayout.addWidget(&m_PresetName, 0, 0);

	// Load preset
	m_LoadPreset.setParent(this);
	m_LoadPreset.setText("Load");
	m_LoadPreset.setToolTip("Load preset");
	m_LoadPreset.setStatusTip("Load transfer function preset");
	m_LoadPreset.setFixedWidth(60);
	m_LoadPreset.setFixedHeight(20);
	m_GridLayout.addWidget(&m_LoadPreset, 0, 1);

	// Save Preset
	m_SavePreset.setParent(this);
	m_SavePreset.setEnabled(false);
	m_SavePreset.setText("Save");
	m_SavePreset.setToolTip("Save Preset");
	m_SavePreset.setStatusTip("Save transfer function preset");
	m_SavePreset.setFixedWidth(60);
	m_SavePreset.setFixedHeight(20);
	m_GridLayout.addWidget(&m_SavePreset, 1, 1);

	// Remove preset
	m_RemovePreset.setParent(this);
	m_RemovePreset.setEnabled(false);
	m_RemovePreset.setText("Remove");
	m_RemovePreset.setToolTip("Remove Preset");
	m_RemovePreset.setStatusTip("Remove transfer function preset");
	m_RemovePreset.setFixedWidth(60);
	m_RemovePreset.setFixedHeight(20);
	m_GridLayout.addWidget(&m_RemovePreset, 2, 1);

	// Load presets
	m_LoadPresets.setParent(this);
	m_LoadPresets.setText("From File");
	m_LoadPresets.setToolTip("Load presets from files");
	m_LoadPresets.setStatusTip("Load transfer function presets from file");
	m_LoadPresets.setFixedWidth(60);
	m_LoadPresets.setFixedHeight(20);
	m_GridLayout.addWidget(&m_LoadPresets, 3, 1);

	// Save presets
	m_SavePresets.setParent(this);
	m_SavePresets.setText("To File");
	m_SavePresets.setToolTip("Save presets to file");
	m_SavePresets.setStatusTip("Save transfer function presets to file");
	m_SavePresets.setFixedWidth(60);
	m_SavePresets.setFixedHeight(20);
	m_GridLayout.addWidget(&m_SavePresets, 4, 1);

	// Dummy
	m_Dummy.setParent(this);
	m_Dummy.setVisible(false);
	m_Dummy.setText("Dummy");
	m_Dummy.setToolTip("Dummy");
	m_Dummy.setStatusTip("Dummy");
	m_Dummy.setFixedWidth(60);
	m_Dummy.setFixedHeight(20);
	m_GridLayout.addWidget(&m_Dummy, 5, 1);
}

void QTransferFunctionPresetsWidget::CreateConnections(void)
{
	connect(&m_LoadPreset, SIGNAL(clicked()), this, SLOT(OnLoadPreset()));
	connect(&m_SavePreset, SIGNAL(clicked()), this, SLOT(OnSavePreset()));
	connect(&m_PresetName, SIGNAL(returnPressed()), this, SLOT(OnSavePreset()));
	connect(&m_RemovePreset, SIGNAL(clicked()), this, SLOT(OnRemovePreset()));
	connect(&m_LoadPresets, SIGNAL(clicked()), this, SLOT(OnLoadPresets()));
	connect(&m_SavePresets, SIGNAL(clicked()), this, SLOT(OnSavePresets()));
	connect(&m_Dummy, SIGNAL(clicked()), this, SLOT(OnDummy()));
	connect(&m_PresetList, SIGNAL(itemSelectionChanged()), this, SLOT(OnPresetSelectionChanged()));
	connect(&m_PresetName, SIGNAL(textChanged(const QString&)), this, SLOT(OnPresetNameChanged(const QString&)));
	connect(&m_PresetList, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(OnPresetItemChanged(QListWidgetItem*)));
}

void QTransferFunctionPresetsWidget::LoadPresetsFromFile(const bool& ChoosePath)
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
		XmlFile.setFileName(QFileDialog::getOpenFileName(this, tr("Load transfer function presets from file"), "", tr("XML Files (*.xml)")));
	}
	else
	{
		XmlFile.setFileName(CurrentPath + "/TransferFunctionPresets.xml");
	}
	
	// Open the XML file
	if (!XmlFile.open(QIODevice::ReadOnly))
	{
		qDebug("Failed to open file for reading.");
		return;
	}

	// Document object model for XML
	QDomDocument DOM("TransferFunctionPresets");

	// Parse file content into DOM
	if (!DOM.setContent(&XmlFile))
	{
		qDebug("Failed to parse the file into a DOM tree.");
		XmlFile.close();
    	return;
	}
 
	// Obtain document root node
	QDomElement DomRoot = DOM.documentElement();

	QDomNodeList Presets = DomRoot.elementsByTagName("Preset");

	for (int i = 0; i < Presets.count(); i++)
	{
		QDomNode TransferFunctionNode = Presets.item(i);

		// Create new transfer function
		QTransferFunction TransferFunction;

		// Append the transfer function
		m_TransferFunctions.append(TransferFunction);

		// Load the preset into it
		m_TransferFunctions.back().ReadXML(TransferFunctionNode.toElement());
	}

	XmlFile.close();

	// Update the presets list
	UpdatePresetsList();
}

void QTransferFunctionPresetsWidget::SavePresetsToFile(const bool& ChoosePath)
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
		XmlFile.setFileName(QFileDialog::getSaveFileName(this, tr("Save transfer function presets to file"), "", tr("XML Files (*.xml)")));
	}
	else
	{
		XmlFile.setFileName(CurrentPath + "/TransferFunctionPresets.xml");
	}
	
	// Open the XML file
	if (!XmlFile.open(QIODevice::WriteOnly ))
	{
		qDebug("Failed to open file for reading.");
		return;
	}

	// Document object model for XML
	QDomDocument DOM;

	// Write each transfer function to the file
	QDomElement DomRoot = DOM.documentElement();

	QDomElement Presets = DOM.createElement("Presets");
	
	for (int i = 0; i < m_TransferFunctions.size(); i++)
	{
		m_TransferFunctions[i].WriteXML(DOM, Presets);
	}

	DOM.appendChild(Presets);

	// Create text stream
	QTextStream TextStream(&XmlFile);

	// Save the XML file
	DOM.save(TextStream, 0);

	// Close the XML file
	XmlFile.close();
}

void QTransferFunctionPresetsWidget::UpdatePresetsList(void)
{
	m_PresetList.clear();

	for (int i = 0; i < m_TransferFunctions.size(); i++)
	{
		// Create new list item
		QPresetItem* pListWidgetItem = new QPresetItem(&m_PresetList, &m_TransferFunctions[i]);

		pListWidgetItem->setFlags(pListWidgetItem->flags() | Qt::ItemIsEditable);

		// Add the item
		m_PresetList.addItem(pListWidgetItem);
	}

	// Get current row
	int CurrentRow = m_PresetList.currentRow();

	m_RemovePreset.setEnabled(CurrentRow >= 0);
	m_LoadPreset.setEnabled(CurrentRow >= 0);
	m_SavePresets.setEnabled(m_PresetList.count() > 0);
}

void QTransferFunctionPresetsWidget::OnPresetSelectionChanged(void)
{
	// Get current row
	int CurrentRow = m_PresetList.currentRow();

	m_LoadPreset.setEnabled(CurrentRow >= 0);
	m_RemovePreset.setEnabled(CurrentRow >= 0);

	if (CurrentRow < 0)
		return;

	m_PresetName.setText(m_PresetList.currentItem()->text());
}

void QTransferFunctionPresetsWidget::OnLoadPreset(void)
{
	// Get current row
	int CurrentRow = m_PresetList.currentRow();
	
	if (CurrentRow < 0)
		return;

	gTransferFunction = m_TransferFunctions.at(CurrentRow);
}

void QTransferFunctionPresetsWidget::OnSavePreset(void)
{
	for (int i = 0; i < m_TransferFunctions.size(); i++)
	{
		if (m_TransferFunctions.at(i).GetName() == m_PresetName.text())
			m_TransferFunctions.removeAt(i);
	}

	QTransferFunction Preset = gTransferFunction;

	Preset.SetName(m_PresetName.text());
	m_TransferFunctions.append(Preset);

	UpdatePresetsList();
}

void QTransferFunctionPresetsWidget::OnRemovePreset(void)
{
	// Get current row
	const int CurrentRow = m_PresetList.currentRow();

	if (CurrentRow < 0)
		return;

	m_TransferFunctions.removeAt(CurrentRow);

	UpdatePresetsList();
}

void QTransferFunctionPresetsWidget::OnLoadPresets(void)
{
	// Clear the current list of presets
	m_TransferFunctions.clear();

	// Load the presets, asking for a location
	LoadPresetsFromFile(true);
}

void QTransferFunctionPresetsWidget::OnSavePresets(void)
{
	// Save the presets, asking for a location
	SavePresetsToFile(true);
}

void QTransferFunctionPresetsWidget::OnDummy(void)
{
	// Save the presets, asking for a location
	SavePresetsToFile(true);
}

void QTransferFunctionPresetsWidget::OnPresetNameChanged(const QString& Text)
{
	m_SavePreset.setEnabled(Text.length() > 0);
}

void QTransferFunctionPresetsWidget::OnPresetItemChanged(QListWidgetItem* pWidgetItem)
{
	QPresetItem* pPresetItem = dynamic_cast<QPresetItem*>(pWidgetItem);

	if (pPresetItem)
		pPresetItem->m_pTransferFunction->SetName(pWidgetItem->text());
}