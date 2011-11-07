/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Preset.h"

class QPresetsWidgetBase : public QGroupBox
{
	Q_OBJECT

public:
	QPresetsWidgetBase(QWidget* pParent, const QString& InternalName, const QString& UserInterfaceName);

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
	void OnLoadPreset(const QString& PresetName);
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
	QPushButton		m_SavePreset;
	QPushButton		m_RenamePreset;
	QPushButton		m_RemovePreset;
	QPushButton		m_LoadPresets;
	QPushButton		m_SavePresets;
	QPresetList		m_PresetItems;
};

template <class T>
class QPresetsWidget : public QPresetsWidgetBase
{
public:
	QPresetsWidget(QWidget* pParent, const QString& InternalName, const QString& UserInterfaceName) :
		QPresetsWidgetBase(pParent, InternalName, UserInterfaceName)
	{
		LoadPresets(false);

		// Add default preset
		InsertPreset(0, T::Default());
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

	void SavePreset(T Preset, const bool& PromptOverwrite = true)
	{
		for (int i = 0; i < m_Presets.size(); i++)
		{
			if (m_Presets[i].GetName() == Preset.GetName())
			{
				if (PromptOverwrite)
				{
					int Result = QMessageBox::question(this, "Preset already exists", "Overwrite?", QMessageBox::Yes | QMessageBox::No);
				
					if (Result == QMessageBox::No)
						return;
				}

				m_Presets[i] = Preset;

				Log(QString("'" + Preset.GetName() + "' " + m_UserInterfaceName.toLower() + " preset saved"));

				return;
			}
		}

		m_Presets.append(Preset);

		// Update GUI
		UpdatePresetsList();

		Log(QString("'" + Preset.GetName() + "' " + m_UserInterfaceName.toLower() + " preset saved"));
	}

	void RemovePreset(void)
	{
		QString PresetName = m_PresetName.currentText();

		if (PresetName == "Default")
		{
			Log("Can't remove 'default' preset");
			return;
		}

// 		if (QMessageBox::question(this, "Remove preset", "Are you sure you want to remove " + PresetName, QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes)
// 			return;

		// Get selected row index
		const int CurrentRow = m_PresetName.currentIndex();

		if (CurrentRow < 0)
			return;

		m_Presets.removeAt(CurrentRow);

		UpdatePresetsList();

		Log(QString("'" + PresetName + "' " + m_UserInterfaceName.toLower() + " preset removed"));
	}

	void LoadPreset(const QString& Name)
	{
		emit QPresetsWidgetBase::LoadPreset(Name);

		Log("Loading '" + Name + "' " + m_UserInterfaceName.toLower() + " preset");
	}

	void LoadPresets(const bool& ChoosePath)
	{
		// Clear presets list
		m_Presets.clear();

		// XML file containing transfer function presets
		QFile XmlFile;

		// File name + extension
		QString FileName = QApplication::applicationDirPath() + "/" + m_InternalName + "Presets.xml";

		Log(QString("Loading " + m_UserInterfaceName.toLower() + " presets from file: "+ QFileInfo(FileName).fileName()).toAscii());

		// Set the file name
		if (ChoosePath)
		{
			// Create open file dialog
			XmlFile.setFileName(GetOpenFileName("Load " + m_UserInterfaceName + " presets from file", "XML Preset Files (*.xml)", "star"));
		}
		else
		{
			XmlFile.setFileName(FileName);
		}

		// Open the XML file for reading
		if (!XmlFile.open(QIODevice::ReadOnly))
		{
			Log(QString("Failed to open " + QFileInfo(FileName).fileName() + " for reading: " + XmlFile.errorString()).toAscii(), QLogger::Critical);
			return;
		}

		// Document object model for XML
		QDomDocument DOM;

		// Parse file content into DOM
		if (!DOM.setContent(&XmlFile))
		{
			Log("Failed to parse " + QFileInfo(FileName).fileName() + ".xml into a DOM tree.", QLogger::Critical);
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

		Log(QString("Saving " + m_UserInterfaceName.toLower() + " presets to file: " + QFileInfo(FileName).fileName()).toAscii());

		// Set the file name
		if (ChoosePath)
		{
			// Create open file dialog
			XmlFile.setFileName(GetSaveFileName("Save " + m_UserInterfaceName + " presets to file", "XML Preset Files (*.xml)", "star"));
		}
		else
		{
			XmlFile.setFileName(FileName);
		}

		// Open the XML file for writing
		if (!XmlFile.open(QIODevice::WriteOnly ))
		{
			Log(QString("Failed to open " + QFileInfo(FileName).fileName() + ".xml for writing: " + XmlFile.errorString()).toAscii(), QLogger::Critical);
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
		{
			if (m_Presets[i].GetName() == Name)
				return;
		}

		QString OldName = m_Presets[Index].GetName();

		// Rename
		m_Presets[Index].SetName(Name);

		// Update GUI
		UpdatePresetsList();

		Log("'" + OldName + "' renamed to '" + Name + "'");
	};

	bool HasPreset(const QString& Name)
	{
		for (int i = 0; i < m_Presets.size(); i++)
		{
			if (m_Presets[i].GetName() == Name)
				return true;
		}

		return false;
	}

	T GetPreset(const QString& Name)
	{
		for (int i = 0; i < m_Presets.size(); i++)
		{
			if (m_Presets[i].GetName() == Name)
				return m_Presets[i];
		}

		return T::Default();
	}

	void LoadPreset(T& PresetDestination, const QString& Name)
	{
		// Only load the preset when it exists
		if (!HasPreset(Name))
			return;

		T SourcePreset = GetPreset(Name);

		if (PresetDestination.GetDirty())
		{
//			int Result = QMessageBox::question(this, "Unsaved changes", "Save changes to " + PresetDestination.GetName() + "?", QMessageBox::Yes | QMessageBox::No);
				
	//		if (Result == QMessageBox::Yes)
		//	{
//				T Preset = PresetDestination;
//				Preset.SetName(Name);

				// Save the preset
//				SavePreset(Preset, false);
		//	}
		}

		// Copy the transfer function
		PresetDestination = SourcePreset;

		// The global transfer function is not dirty
		PresetDestination.SetDirty(false);
	};

private:
	QList<T>	m_Presets;
};