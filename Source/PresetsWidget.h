#pragma once

#include <QtGui>
#include <QtXml\qdom.h>

#include "TransferFunction.h"

class QPresetItem : public QListWidgetItem
{
public:
	QPresetItem(QListWidget* pListWidget, const QString& Name, void* pData) :
		QListWidgetItem(pListWidget),
		m_pData(pData)
	{
		setText(Name);
	}

	void* m_pData;
};

template <class T>
class QPresetsWidget : public QGroupBox
{
    Q_OBJECT

public:
	QPresetsWidget(const QString& PresetFileName, QWidget* pParent /*= NULL*/ ) :
		QGroupBox(pParent),
		m_PresetFileName(PresetFileName),
		m_MainLayout(),
		m_PresetName(),
		m_LoadPreset(),
		m_SavePreset(),
		m_RemovePreset(),
		m_LoadPresets(),
		m_SavePresets(),
		m_PresetItems()
	{
		// Title, status and tooltip
		setTitle("Presets");
		setToolTip("Presets");
		setStatusTip("Presets");

		m_PresetFileName = PresetFileName;

		CreateUI();
		CreateConnections();
	}

	void CreateUI(void)
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

	void CreateConnections(void)
	{
		connect(&m_LoadPreset, SIGNAL(clicked()), this, SLOT(OnLoadPreset()));
		connect(&m_SavePreset, SIGNAL(clicked()), this, SLOT(OnSavePreset()));
		connect(&m_RemovePreset, SIGNAL(clicked()), this, SLOT(OnRemovePreset()));
		connect(&m_LoadPresets, SIGNAL(clicked()), this, SLOT(OnLoadPresets()));
		connect(&m_SavePresets, SIGNAL(clicked()), this, SLOT(OnSavePresets()));
		connect(&m_PresetName, SIGNAL(editTextChanged(const QString&)), this, SLOT(OnPresetNameChanged(const QString&)));

		connect(qApp, SIGNAL(aboutToQuit ()), this, SLOT(OnApplicationAboutToExit()));
	}

	void LoadPresetsFromFile(const bool& ChoosePath)
	{
		return;

		qDebug("Loading presets from file");

		// XML file containing transfer function presets
		QFile XmlFile;

		// Get applications working directory
		QString CurrentPath = QDir::currentPath();

		// Set the file name
		if (ChoosePath)
		{
			// Create open file dialog
			XmlFile.setFileName(QFileDialog::getOpenFileName(this, "Load preset from file", "", tr("XML Files (*.xml)")));
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
		QDomDocument DOM;

		// Parse file content into DOM
		if (!DOM.setContent(&XmlFile))
		{
			qDebug("Failed to parse file into a DOM tree.");
			XmlFile.close();
    		return;
		}
 
		// Obtain document root node
		QDomElement DomRoot = DOM.documentElement();

		this->LoadPresets(DomRoot);
	
		/*
		QDomNodeList Presets = Root.elementsByTagName("Preset");

		for (int i = 0; i < Presets.count(); i++)
		{
			QDomNode TransferFunctionNode = Presets.item(i);

			// Create new transfer function
			QTransferFunction* pTransferFunction = new QTransferFunction();

			// Append the transfer function
			m_PresetItems.append(pTransferFunction);

			// Load the preset into it
			m_PresetItems.back()->ReadXML(TransferFunctionNode.toElement());
		}
		*/

		XmlFile.close();

		// Update the presets list
		UpdatePresetsList();
	}

	void SavePresetsToFile(const bool& ChoosePath)
	{
		return;

		qDebug("Saving presets to file");

		return;

		// XML file containing transfer function presets
		QFile XmlFile;

		// Get applications working directory
		QString CurrentPath = QDir::currentPath();

		// Set the file name
		if (ChoosePath)
		{
			// Create open file dialog
			XmlFile.setFileName(QFileDialog::getSaveFileName(this, "Save preset to file", "", tr("XML Files (*.xml)")));
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
		QDomDocument DOM;

		// Write each transfer function to the file
		QDomElement DomRoot = DOM.documentElement();

		this->SavePresets(DOM, DomRoot);

		// Create text stream
		QTextStream TextStream(&XmlFile);

		// Save the XML file
		DOM.save(TextStream, 0);

		/*
		QDomElement Presets = DomDoc.createElement("Presets");

		for (int i = 0; i < m_PresetItems.size(); i++)
		{
			m_PresetItems[i]->WriteXML(DomDoc, Presets);
		}

		Root.appendChild(Presets);
		*/

		// Close the XML file
		XmlFile.close();
	}

	void LoadPresets(QDomElement& Root)
	{
		this->LoadPresets(Root);
	}

	void SavePresets(QDomDocument& DomDoc, QDomElement& Root)
	{
		SavePresets(DomDoc, Root);
	}

	void LoadPreset(QPresetXML* pPreset)
	{
		LoadPreset(pPreset);
	}

	void SavePreset(const QString& Name)
	{
		SavePreset(Name);
	}

	void UpdatePresetsList(void)
	{
		for (int i = 0; i < m_PresetItems.size(); i++)
		{
			// Put pointer to preset in void pointer
			QVariant Variant = qVariantFromValue((void*)m_PresetItems[i]);

			m_PresetName.addItem(m_PresetItems[i]->GetName(), Variant);
		}
	}

	void OnLoadPreset(void)
	{
		if (m_PresetName.currentIndex() < 0)
			return;

		QVariant Variant = m_PresetName.itemData(m_PresetName.currentIndex());

		this->LoadPreset((QPresetXML*)Variant.value<void*>());
	}

	void OnSavePreset(void)
	{
		this->SavePreset(m_PresetName.lineEdit()->text());
	}

	void OnRemovePreset(void)
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

	void OnLoadPresets(void)
	{
		// Clear the current list of presets
	//	m_TransferFunctions.clear();

		// Load the presets, asking for a location
		LoadPresetsFromFile(true);
	}

	void OnSavePresets(void)
	{
		// Save the presets, asking for a location
		SavePresetsToFile(true);
	}

	void OnDummy(void)
	{
		// Save the presets, asking for a location
		SavePresetsToFile(true);
	}

	void OnPresetNameChanged(const QString& Text)
	{
		m_SavePreset.setEnabled(Text.length() > 0);
	}

	void OnPresetItemChanged(QListWidgetItem* pWidgetItem)
	{
		QPresetItem* pPresetItem = dynamic_cast<QPresetItem*>(pWidgetItem);
		//(YourClass *) v.value<void *>();
	//	if (pPresetItem)
	//		((QTransferFunction*)pPresetItem->m_pData)->SetName(pWidgetItem->text());
	}

	void OnApplicationAboutToExit(void)
	{
		SavePresetsToFile();
	}

	virtual QSize sizeHint() const { return QSize(10, 10); }

protected slots:

protected:
	QString			m_PresetFileName;
	QGridLayout		m_MainLayout;
	QComboBox		m_PresetName;
	QPushButton		m_LoadPreset;
	QPushButton		m_SavePreset;
	QPushButton		m_RemovePreset;
	QPushButton		m_LoadPresets;
	QPushButton		m_SavePresets;
	QPresetList		m_PresetItems;
	QList<T>		m_Presets;
};