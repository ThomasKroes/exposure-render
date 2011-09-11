#pragma once

#include "Preset.h"
#include "Controls.h"

class QTestWidget : public QGroupBox
{
	Q_OBJECT

public:
	QTestWidget(QWidget* pParent, const QString& InternalName, const QString& UserInterfaceName);

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
	void OnLoadPreset(void);
	void OnSavePreset(void);
	void OnRenamePreset(void);
	void OnRemovePreset(void);
	void OnSavePresets(void);
	void OnLoadPresets(void);
	void OnCurrentIndexChanged(int Index);

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

	void SavePreset(T Preset)
	{
		for (int i = 0; i < m_Presets.size(); i++)
		{
			if (m_Presets[i].GetName() == Preset.GetName())
			{
				int Result = QMessageBox::question(this, "Preset already exists", "Overwrite?", QMessageBox::Yes | QMessageBox::No);
				
				if (Result == QMessageBox::No)
					return;

				m_Presets[i] = Preset;
				return;
			}
		}

		m_Presets.append(Preset);

		// Update GUI
		UpdatePresetsList();
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

	void LoadPreset(const QString& Name)
	{
		emit QTestWidget::LoadPreset(Name);
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
		QString FileName = QApplication::applicationDirPath() + "/" + m_InternalName + "Presets.xml";

		qDebug(QString("Loading: " + QFileInfo(FileName).fileName()).toAscii());

		// Set the file name
		if (ChoosePath)
		{
			// Create open file dialog
			XmlFile.setFileName(GetOpenFileName("Load " + m_UserInterfaceName + " presets from file", "XML Preset Files (*.xml)"));
		}
		else
		{
			XmlFile.setFileName(FileName);
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
		QString FileName = QApplication::applicationDirPath() + "/" + m_InternalName + "Presets.xml";

		qDebug(QString("Saving: " + QFileInfo(FileName).fileName()).toAscii());

		// Set the file name
		if (ChoosePath)
		{
			// Create open file dialog
			XmlFile.setFileName(GetSaveFileName("Save " + m_UserInterfaceName + " presets to file", "XML Preset Files (*.xml)"));
		}
		else
		{
			XmlFile.setFileName(FileName);
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
		{
			if (m_Presets[i].GetName() == "Default")
				continue;

			m_Presets[i].WriteXML(DOM, Presets);
		}

		DOM.appendChild(Presets);

		// Create text stream
		QTextStream TextStream(&XmlFile);

		// Save the XML file
		DOM.save(TextStream, 0);

		// Close the XML file
		XmlFile.close();
	};
	
	void AddPreset(T& Preset)
	{
		m_Presets.append(Preset);

		// Update GUI
		UpdatePresetsList();
	};

	void InsertPreset(const int& Index, T& Preset)
	{
		m_Presets.insert(Index, Preset);

		// Update GUI
		UpdatePresetsList();
	};

	void RenamePreset(const int& Index, const QString& Name)
	{
		// Check if preset with same name already exists
		for (int i = 0; i < m_Presets.size(); i++)
			if (m_Presets[i].GetName() == Name)
				return;

		// Rename
		m_Presets[Index].SetName(Name);

		// Update GUI
		UpdatePresetsList();
	};

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

private:
	QList<T>	m_Presets;
};