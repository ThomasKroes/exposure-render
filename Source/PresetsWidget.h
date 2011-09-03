#pragma once

#include <QtGui>
#include <QtXml\qdom.h>

#include "Preset.h"

class QTestWidget : public QGroupBox
{
	Q_OBJECT

public:
	QTestWidget(QWidget* pParent, const QString& InternalName, const QString& UserInterfaceName) :
		QGroupBox(pParent),
		m_InternalName(InternalName),
		m_UserInterfaceName(UserInterfaceName)
	{
		// Title, status and tooltip
		setTitle("Presets");
		setToolTip(UserInterfaceName + " presets");
		setStatusTip(UserInterfaceName + " presets");

		// Assign layout
		setLayout(&m_MainLayout);

		// Name edit
		m_PresetName.setEditable(true);
		m_MainLayout.addWidget(&m_PresetName, 0, 0);

		// Load preset
		m_LoadPreset.setText("L");
		m_LoadPreset.setToolTip("Load " + m_UserInterfaceName.toLower() + " preset");
		m_LoadPreset.setStatusTip("Load " + m_UserInterfaceName.toLower() + " preset");
		m_LoadPreset.setFixedWidth(20);
		m_LoadPreset.setFixedHeight(20);
		m_MainLayout.addWidget(&m_LoadPreset, 0, 1);

		// Save Preset
		m_SavePreset.setText("S");
		m_SavePreset.setToolTip("Save " + m_UserInterfaceName.toLower() + " Preset");
		m_SavePreset.setStatusTip("Save " + m_UserInterfaceName.toLower() + " preset");
		m_SavePreset.setFixedWidth(20);
		m_SavePreset.setFixedHeight(20);
		m_MainLayout.addWidget(&m_SavePreset, 0, 2);

		// Rename Preset
		m_RenamePreset.setText("S");
		m_RenamePreset.setToolTip("Rename " + m_UserInterfaceName.toLower() + " Preset");
		m_RenamePreset.setStatusTip("Rename " + m_UserInterfaceName.toLower() + " preset");
		m_RenamePreset.setFixedWidth(20);
		m_RenamePreset.setFixedHeight(20);
		m_MainLayout.addWidget(&m_RenamePreset, 0, 3);

		// Remove preset
		m_RemovePreset.setText("R");
		m_RemovePreset.setToolTip("Remove " + m_UserInterfaceName.toLower() + " Preset");
		m_RemovePreset.setStatusTip("Remove " + m_UserInterfaceName.toLower() + " preset");
		m_RemovePreset.setFixedWidth(20);
		m_RemovePreset.setFixedHeight(20);
		m_MainLayout.addWidget(&m_RemovePreset, 0, 4);

		// Load presets
		m_LoadPresets.setText("LF");
		m_LoadPresets.setToolTip("Load " + m_UserInterfaceName.toLower() + " presets from file");
		m_LoadPresets.setStatusTip("Load " + m_UserInterfaceName.toLower() + " presets from file");
		m_LoadPresets.setFixedWidth(20);
		m_LoadPresets.setFixedHeight(20);
		m_MainLayout.addWidget(&m_LoadPresets, 0, 5);

		// Save presets
		m_SavePresets.setText("SF");
		m_SavePresets.setToolTip("Save " + m_UserInterfaceName.toLower() + " presets to file");
		m_SavePresets.setStatusTip("Save " + m_UserInterfaceName.toLower() + " presets to file");
		m_SavePresets.setFixedWidth(20);
		m_SavePresets.setFixedHeight(20);
		m_MainLayout.addWidget(&m_SavePresets, 0, 6);

		// Connections
		connect(&m_LoadPreset, SIGNAL(clicked()), this, SLOT(OnLoadPreset()));
		connect(&m_SavePreset, SIGNAL(clicked()), this, SLOT(OnSavePreset()));
		connect(&m_RenamePreset, SIGNAL(clicked()), this, SLOT(OnRenamePreset()));
		connect(&m_RemovePreset, SIGNAL(clicked()), this, SLOT(OnRemovePreset()));
		connect(&m_LoadPresets, SIGNAL(clicked()), this, SLOT(OnLoadPresets()));
		connect(&m_SavePresets, SIGNAL(clicked()), this, SLOT(OnSavePresets()));
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

	void OnRenamePreset(void)
	{
		if (m_PresetName.currentIndex() < 0 || m_PresetName.lineEdit()->text().length() <= 0)
			return;

		QString Name = QInputDialog::getText(this, "Rename Preset", "Name", QLineEdit::Normal, m_PresetName.lineEdit()->text());

		this->RenamePreset(m_PresetName.currentIndex(), Name);
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
	QString			m_InternalName;
	QString			m_UserInterfaceName;
	QGridLayout		m_MainLayout;
	QComboBox		m_PresetName;
	QPushButton		m_LoadPreset;
	QPushButton		m_SavePreset;
	QPushButton		m_RenamePreset;
	QPushButton		m_RemovePreset;
	QPushButton		m_LoadPresets;
	QPushButton		m_SavePresets;
	QPresetList		m_PresetItems;
};

template <class T>
class QPresetsWidget : public QTestWidget
{
public:
	QPresetsWidget(QWidget* pParent, const QString& InternalName, const QString& UserInterfaceName) :
		QTestWidget(pParent, InternalName, UserInterfaceName)
	{
		LoadPresets(false);
	}

	virtual ~QPresetsWidget(void)
	{
		SavePresets(false);
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
		// Clear presets list
		m_Presets.clear();

		// XML file containing transfer function presets
		QFile XmlFile;

		// Get applications working directory
		QString CurrentPath = QDir::currentPath();

		// File name + extension
		QString FileName = m_InternalName + "Presets.xml";

		qDebug(QString("Loading " + m_UserInterfaceName + " presets from file: " + CurrentPath + "/" + FileName).toAscii());

		// Set the file name
		if (ChoosePath)
		{
			// Create open file dialog
			XmlFile.setFileName(QFileDialog::getOpenFileName(this, "Load " + m_UserInterfaceName + " preset from file", "", tr("XML Files (*.xml)")));
		}
		else
		{
			XmlFile.setFileName(CurrentPath + "/" + FileName);
		}

		// Open the XML file for reading
		if (!XmlFile.open(QIODevice::ReadOnly))
		{
			qDebug(QString("Failed to open file for reading: " + XmlFile.errorString()).toAscii());
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
		// XML file containing transfer function presets
		QFile XmlFile;

		// Get applications working directory
		QString CurrentPath = QDir::currentPath();

		// File name + extension
		QString FileName = m_InternalName + "Presets.xml";

		qDebug(QString("Saving " + m_UserInterfaceName + " presets to file: " + CurrentPath + "/" + FileName).toAscii());

		// Set the file name
		if (ChoosePath)
		{
			// Create open file dialog
			XmlFile.setFileName(QFileDialog::getSaveFileName(this, "Save " + m_UserInterfaceName + " preset to file", "", tr("XML Files (*.xml)")));
		}
		else
		{
			XmlFile.setFileName(CurrentPath + "/" + FileName);
		}

		// Open the XML file for writing
		if (!XmlFile.open(QIODevice::WriteOnly ))
		{
			qDebug(QString("Failed to open file for writing: " + XmlFile.errorString()).toAscii());
			return;
		}

		// Document object model for XML
		QDomDocument DOM(m_InternalName);

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
	
	void RenamePreset(const int& Index, const QString& Name)
	{
		// Rename
		m_Presets[Index].SetName(Name);

		// Update GUI
		UpdatePresetsList();
	};

	void AddPreset(T Preset)
	{
		m_Presets.append(Preset);

		// Update GUI
		UpdatePresetsList();
	}

	T GetPreset(const QString& Name)
	{
		for (int i = 0; i < m_Presets.size(); i++)
		{
			if (m_Presets[i].GetName() == Name)
				return m_Presets[i];
		}

		T Preset;
		return Preset;
	}

	QList<T>	m_Presets;
};