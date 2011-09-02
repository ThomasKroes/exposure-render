#pragma once

#include <QtGui>
#include <QtXml\qdom.h>

#include "Preset.h"
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
class QPresets
{
public:
	void Add(T Preset)
	{
		m_Presets.append(Preset);
	}

	void Remove(T Preset)
	{
		m_Presets.remove(Preset);
	}

	void LoadPresetsFromFile(const bool& ChoosePath = false)
	{

	}

	void SavePresetsFromFile(const bool& ChoosePath = false)
	{

	}

protected slots:

protected:
	QList<T>		m_Presets;
};

class QPresetsWidget : public QGroupBox
{
    Q_OBJECT

public:
	QPresetsWidget(const QString& PresetFileName = "", QWidget* pParent = NULL);

	void CreateUI(void);

	void CreateConnections(void);

	void LoadPresetsFromFile(const bool& ChoosePath = false);

	void SavePresetsToFile(const bool& ChoosePath = false);

	void LoadPresets(QDomElement& Root);

	void SavePresets(QDomDocument& DomDoc, QDomElement& Root);

	void LoadPreset(QPresetXML* pPreset);

	void SavePreset(const QString& Name);

	void UpdatePresetsList(void);

	void OnLoadPreset(void);

	void OnSavePreset(void);

	void OnRemovePreset(void);

	void OnLoadPresets(void);

	void OnSavePresets(void);

	void OnPresetNameChanged(const QString& Text);

	void OnPresetItemChanged(QListWidgetItem* pWidgetItem);

	void OnApplicationAboutToExit(void);

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
};

class QTestWidget : public QGroupBox
{
	Q_OBJECT

public:
	QTestWidget(void)
	{
		// Title, status and tooltip
		setTitle("Presets");
		setToolTip("Presets");
		setStatusTip("Presets");

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
		m_SavePreset.setText("S");
		m_SavePreset.setToolTip("Save Preset");
		m_SavePreset.setStatusTip("Save transfer function preset");
		m_SavePreset.setFixedWidth(20);
		m_SavePreset.setFixedHeight(20);
		m_MainLayout.addWidget(&m_SavePreset, 0, 2);

		// Remove preset
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

		// Connections
		connect(&m_LoadPreset, SIGNAL(clicked()), this, SLOT(OnLoadPreset()));
		connect(&m_SavePreset, SIGNAL(clicked()), this, SLOT(OnSavePreset()));
		connect(&m_RemovePreset, SIGNAL(clicked()), this, SLOT(OnRemovePreset()));
		connect(&m_LoadPresets, SIGNAL(clicked()), this, SLOT(OnLoadPresets()));
		connect(&m_SavePresets, SIGNAL(clicked()), this, SLOT(OnSavePresets()));
		connect(&m_PresetName, SIGNAL(editTextChanged(const QString&)), this, SLOT(OnEditTextChanged(const QString&)));
		connect(&m_PresetName, SIGNAL(currentIndexChanged(int)), this, SLOT(OnCurrentIndexChanged(int)));
	}

	virtual void RemovePreset(void)
	{
	};

	virtual void LoadPresets(const bool& ChoosePath)
	{
	};

	virtual void SavePresets(const bool& ChoosePath)
	{
	};

	virtual void RenamePreset(const int& Index, const QString& Name)
	{
	};

public slots:
	void OnLoadPreset(void)
	{
		if (m_PresetName.lineEdit()->text().length() <= 0)
			return;

		emit LoadPreset(m_PresetName.lineEdit()->text());
	}

	void OnSavePreset(void)
	{
		if (m_PresetName.lineEdit()->text().length() <= 0)
			return;

		emit SavePreset(m_PresetName.lineEdit()->text());
	}

	void OnRemovePreset(void)
	{
		this->RemovePreset();
	}

	void OnSavePresets(void)
	{
		this->SavePresets(true);
	}

	void OnLoadPresets(void)
	{
		this->LoadPresets(true);
	}

	void OnEditTextChanged(const QString& Name)
	{
		if (m_PresetName.currentIndex() <= 0)
			return;

		this->RenamePreset(m_PresetName.currentIndex(), Name);
	}

	void OnCurrentIndexChanged(int Index)
	{
		if (Index <= 0)
			return;

		if (m_PresetName.lineEdit()->text().length() > 0)
		{
			m_LoadPreset.setEnabled(true);
			m_SavePreset.setEnabled(true);
		}
	}

signals:
	void LoadPreset(const QString& Name);
	void SavePreset(const QString& Name);

protected:
	QGridLayout		m_MainLayout;
	QComboBox		m_PresetName;
	QPushButton		m_LoadPreset;
	QPushButton		m_SavePreset;
	QPushButton		m_RemovePreset;
	QPushButton		m_LoadPresets;
	QPushButton		m_SavePresets;
	QPresetList		m_PresetItems;
};

template <class T>
class QTemplateWidget : public QTestWidget
{
public:
	QTemplateWidget(const QString& Name) :
		QTestWidget(),
		m_Name(Name)
	{
	}

	void UpdatePresetsList(void)
	{
		// Clear the combobox
		m_PresetName.clear();

		for (int i = 0; i < m_Presets.size(); i++)
		{
			// Put pointer to preset in void pointer of variant
			QVariant Variant = qVariantFromValue((void*)m_PresetItems[i]);

			// Add the item
			m_PresetName.addItem(m_Presets[i].GetName(), Variant);
		}
	}

	void RemovePreset(void)
	{
		// Get selected row index
		const int CurrentRow = m_PresetName.currentIndex();

		if (CurrentRow < 0)
			return;

		m_Presets.removeAt(CurrentRow);

		UpdatePresetsList();
	}

	void LoadPresets(const bool& ChoosePath)
	{
		qDebug("QTemplateWidget::LoadPresets");

		// Clear presets list
		m_Presets.clear();

		// XML file containing transfer function presets
		QFile XmlFile;

		// Get applications working directory
		QString CurrentPath = QDir::currentPath();

		// File name + extension
		QString FileName = m_Name + ".xml";

		// Set the file name
		if (ChoosePath)
		{
			// Create open file dialog
			XmlFile.setFileName(QFileDialog::getOpenFileName(this, "Load preset from file", "", tr("XML Files (*.xml)")));
		}
		else
		{
			XmlFile.setFileName(CurrentPath + "/" + FileName);
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
		QDomElement Root = DOM.documentElement();

		QDomNodeList Presets = Root.elementsByTagName("Preset");

		for (int i = 0; i < Presets.count(); i++)
		{
			QDomNode Node = Presets.item(i);

			T NewPreset;

			// Append the transfer function
			m_Presets.append(NewPreset);

			// Load the preset into it
			m_Presets.back().ReadXML(Node.toElement());
		}

		XmlFile.close();

		// Update the combobox to reflect the changes
		UpdatePresetsList();
	}

	void SavePresets(const bool& ChoosePath)
	{
		qDebug("QTemplateWidget::SavePresets");

		// XML file containing transfer function presets
		QFile XmlFile;

		// Get applications working directory
		QString CurrentPath = QDir::currentPath();

		// File name + extension
		QString FileName = m_Name + ".xml";

		// Set the file name
		if (ChoosePath)
		{
			// Create open file dialog
			XmlFile.setFileName(QFileDialog::getSaveFileName(this, "Save preset to file", "", tr("XML Files (*.xml)")));
		}
		else
		{
			XmlFile.setFileName(CurrentPath + "/" + FileName);
		}

		// Open the XML file
		if (!XmlFile.open(QIODevice::WriteOnly ))
		{
			qDebug("Failed to open file for writing.");
			return;
		}

		// Document object model for XML
		QDomDocument DOM("Presets");

		// Document root
		QDomElement Root = DOM.documentElement();

		// Create root element
		QDomElement Presets = DOM.createElement("Presets");

		// Write
		for (int i = 0; i < m_Presets.size(); i++)
			m_Presets[i].WriteXML(DOM, Presets);

		DOM.appendChild(Presets);

		// Create text stream
		QTextStream TextStream(&XmlFile);

		// Save the XML file
		DOM.save(TextStream, 0);

		// Close the XML file
		XmlFile.close();
	};
	
	virtual void RenamePreset(const int& Index, const QString& Name)
	{
		// Rename
		m_Presets[Index].SetName(Name);

		// Update GUI
		UpdatePresetsList();
	};

	void SavePreset(T Preset)
	{
		for (int i = 0; i < m_Presets.size(); i++)
		{
			if (Preset.GetName() == m_Presets[i].GetName())

		}
	}

	T GetPreset(void)
	{
		T Test;
		return Test;
	}

	QString		m_Name;
	QList<T>	m_Presets;
};