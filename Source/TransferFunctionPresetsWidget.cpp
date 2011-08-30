
#include "TransferFunctionPresetsWidget.h"

QTransferFunctionPresetsWidget::QTransferFunctionPresetsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_PresetName(),
	m_LoadPreset(),
	m_SavePreset(),
	m_RemovePreset(),
	m_RenamePreset(),
	m_LoadPresets(),
	m_SavePresets(),
	m_PresetList(),
	m_TransferFunctions()
{
	// Title, status and tooltip
	setTitle("Transfer Function Presets");
	setToolTip("Transfer Function Presets");
	setStatusTip("Transfer Function Presets");

//	setFixedHeight(150);
	setFixedWidth(250);

	// Create grid layout
	m_GridLayout.setAlignment(Qt::AlignTop);

	// Assign layout
	setLayout(&m_GridLayout);

	// Presets list
	m_PresetList.setParent(this);
	m_PresetList.setSelectionMode(QAbstractItemView::SingleSelection);
	m_PresetList.setAlternatingRowColors(true);
	m_PresetList.addItem("Bonsai");
	m_PresetList.addItem("Engine");
	m_PresetList.addItem("Manix");
	m_GridLayout.addWidget(&m_PresetList, 0, 0, 1, 7);

	// Name edit
	m_PresetName.setParent(this);
	m_GridLayout.addWidget(&m_PresetName, 1, 0);

	// Load preset
	m_LoadPreset.setParent(this);
	m_LoadPreset.setText("L");
	m_LoadPreset.setToolTip("Load preset");
	m_LoadPreset.setStatusTip("Load transfer function preset");
	m_LoadPreset.setFixedWidth(20);
	m_LoadPreset.setFixedHeight(20);
	m_GridLayout.addWidget(&m_LoadPreset, 1, 1);

	// Save Preset
	m_SavePreset.setParent(this);
	m_SavePreset.setText("S");
	m_SavePreset.setToolTip("Save Preset");
	m_SavePreset.setStatusTip("Save transfer function preset");
	m_SavePreset.setFixedWidth(20);
	m_SavePreset.setFixedHeight(20);
	m_GridLayout.addWidget(&m_SavePreset, 1, 2);

	// Remove preset
	m_RemovePreset.setParent(this);
	m_RemovePreset.setText("R");
	m_RemovePreset.setToolTip("Remove Preset");
	m_RemovePreset.setStatusTip("Remove transfer function preset");
	m_RemovePreset.setFixedWidth(20);
	m_RemovePreset.setFixedHeight(20);
	m_GridLayout.addWidget(&m_RemovePreset, 1, 3);

	// Rename preset
	m_RenamePreset.setParent(this);
	m_RenamePreset.setText("R");
	m_RenamePreset.setToolTip("Rename Preset");
	m_RenamePreset.setStatusTip("Rename transfer function preset");
	m_RenamePreset.setFixedWidth(20);
	m_RenamePreset.setFixedHeight(20);
	m_GridLayout.addWidget(&m_RenamePreset, 1, 4);

	// Load presets
	m_LoadPresets.setParent(this);
	m_LoadPresets.setText("L");
	m_LoadPresets.setToolTip("Load presets from files");
	m_LoadPresets.setStatusTip("Load transfer function presets from file");
	m_LoadPresets.setFixedWidth(20);
	m_LoadPresets.setFixedHeight(20);
	m_GridLayout.addWidget(&m_LoadPresets, 1, 5);

	// Save presets
	m_SavePresets.setParent(this);
	m_SavePresets.setText("S");
	m_SavePresets.setToolTip("Save presets to file");
	m_SavePresets.setStatusTip("Save transfer function presets to file");
	m_SavePresets.setFixedWidth(20);
	m_SavePresets.setFixedHeight(20);
	m_GridLayout.addWidget(&m_SavePresets, 1, 6);

	// Load transfer function presets from file
	LoadPresetsFromFile();

	// Setup connections
	connect(&m_LoadPreset, SIGNAL(clicked()), this, SLOT(OnLoadPreset()));
	connect(&m_SavePreset, SIGNAL(clicked()), this, SLOT(OnSavePreset()));
	connect(&m_RemovePreset, SIGNAL(clicked()), this, SLOT(OnRemovePreset()));
	connect(&m_RenamePreset, SIGNAL(clicked()), this, SLOT(OnRenamePreset()));
	connect(&m_LoadPresets, SIGNAL(clicked()), this, SLOT(OnLoadPresets()));
	connect(&m_SavePresets, SIGNAL(clicked()), this, SLOT(OnSavePresets()));
}

QTransferFunctionPresetsWidget::~QTransferFunctionPresetsWidget(void)
{
	// Save transfer function presets to file
	SavePresetsToFile();
}

void QTransferFunctionPresetsWidget::LoadPresetsFromFile(void)
{
	// XML file containing transfer function presets
	QFile XmlFile;

	// Get applications working directory
	QString CurrentPath = QDir::currentPath();

	// Set the file name
	XmlFile.setFileName(CurrentPath + "/TransferFunctionPresets.xml");
	
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

		// Load the preset into it
		TransferFunction.ReadXML(TransferFunctionNode.toElement());

		// Append the transfer function
		m_TransferFunctions.append(TransferFunction);
	}

	XmlFile.close();

	// Update the presets list
	UpdatePresetsList();
}

void QTransferFunctionPresetsWidget::SavePresetsToFile(void)
{
	// XML file containing transfer function presets
	QFile XmlFile;

	// Get applications working directory
	QString CurrentPath = QDir::currentPath();

	// Set the file name
	XmlFile.setFileName(CurrentPath + "/TransferFunctionPresets.xml");
	
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
	
	foreach (QTransferFunction TF, m_TransferFunctions)
	{
		TF.WriteXML(DOM, Presets);
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

	foreach (QTransferFunction TransferFunction, m_TransferFunctions)
	{
		// Create new list item
		QListWidgetItem* pListWidgetItem = new QListWidgetItem(TransferFunction.GetName());

		// Add the item
		m_PresetList.addItem(pListWidgetItem);
	}
}

void QTransferFunctionPresetsWidget::OnLoadPreset(void)
{
	// Get current row
	int CurrentRow = m_PresetList.currentRow();
	
	if (CurrentRow < 0)
		return;

	gTransferFunction = m_TransferFunctions[CurrentRow];
}

void QTransferFunctionPresetsWidget::OnSavePreset(void)
{
}

void QTransferFunctionPresetsWidget::OnRemovePreset(void)
{
}

void QTransferFunctionPresetsWidget::OnRenamePreset(void)
{
}

void QTransferFunctionPresetsWidget::OnLoadPresets(void)
{
}


void QTransferFunctionPresetsWidget::OnSavePresets(void)
{
}

