
#include "TransferFunctionPresetsWidget.h"

QTransferFunctionPresetsWidget::QTransferFunctionPresetsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_PresetName(),
	m_LoadPresets(),
	m_SavePresets(),
	m_SavePreset(),
	m_RemovePreset(),
	m_PresetList(),
	m_TransferFunctions(),
	m_Model(100, 1),
	m_DOM("TransferFunctionPresets")
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
	m_PresetList.setCaption("asdasd");
	m_PresetList.setAlternatingRowColors(true);
	m_PresetList.addItem("Bonsai");
	m_PresetList.addItem("Engine");
	m_PresetList.addItem("Manix");
	m_GridLayout.addWidget(&m_PresetList, 0, 0, 1, 5);

	// Name edit
	m_PresetName.setParent(this);
	m_GridLayout.addWidget(&m_PresetName, 1, 0);

	// Load presets
	m_LoadPresets.setParent(this);
	m_LoadPresets.setText("L");
	m_LoadPresets.setToolTip("Load presets from files");
	m_LoadPresets.setStatusTip("Load transfer function presets from file");
	m_LoadPresets.setFixedWidth(20);
	m_LoadPresets.setFixedHeight(20);
	m_GridLayout.addWidget(&m_LoadPresets, 1, 1);

	// Save presets
	m_SavePresets.setParent(this);
	m_SavePresets.setText("S");
	m_SavePresets.setToolTip("Save presets to file");
	m_SavePresets.setStatusTip("Save transfer function presets to file");
	m_SavePresets.setFixedWidth(20);
	m_SavePresets.setFixedHeight(20);
	m_GridLayout.addWidget(&m_SavePresets, 1, 2);

	// Save Preset
	m_SavePreset.setParent(this);
	m_SavePreset.setText("S");
	m_SavePreset.setToolTip("Save Preset");
	m_SavePreset.setStatusTip("Save transfer function preset");
	m_SavePreset.setFixedWidth(20);
	m_SavePreset.setFixedHeight(20);
	m_GridLayout.addWidget(&m_SavePreset, 1, 3);

	// Remove preset
	m_RemovePreset.setParent(this);
	m_RemovePreset.setText("R");
	m_RemovePreset.setToolTip("Remove Preset");
	m_RemovePreset.setStatusTip("Remove transfer function preset");
	m_RemovePreset.setFixedWidth(20);
	m_RemovePreset.setFixedHeight(20);
	m_GridLayout.addWidget(&m_RemovePreset, 1, 4);

	m_TransferFunctions.append(QTransferFunction(this, "Default"));

	// Load transfer function presets from file
	LoadPresetsFromFile();
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

	// Parse file content into DOM
	if (!m_DOM.setContent(&XmlFile))
	{
		qDebug("Failed to parse the file into a DOM tree.");
		XmlFile.close();
    	return;
	}
 
	// Obtain document root node
	QDomElement DomRoot = m_DOM.documentElement();

	QDomNodeList PresetNodes = DomRoot.childNodes();

	for (int i = 0; i < PresetNodes.count(); i++)
	{
		QDomNode Node = PresetNodes.item(i);

		// Create new transfer function
		QTransferFunction TransferFunction;

		// Load the preset into it
		TransferFunction.ReadXML(Node.toElement());
	}

	XmlFile.close();
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

	// Clear contents of DOM
	m_DOM.clear();

	// Write each transfer function to the file
	QDomElement DomRoot = m_DOM.documentElement();

	QDomElement Presets = m_DOM.createElement("Presets");
	
	foreach (QTransferFunction TF, m_TransferFunctions)
	{
		TF.WriteXML(m_DOM, Presets);
	}

	m_DOM.appendChild(Presets);

	// Create text stream
	QTextStream TextStream(&XmlFile);

	// Save the XML file
	m_DOM.save(TextStream, 0);

	// Close the XML file
	XmlFile.close();
}