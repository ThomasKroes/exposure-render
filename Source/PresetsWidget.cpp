/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "PresetsWidget.h"

QPresetsWidgetBase::QPresetsWidgetBase(QWidget* pParent, const QString& InternalName, const QString& UserInterfaceName) :
	QGroupBox(pParent),
	m_InternalName(InternalName),
	m_UserInterfaceName(UserInterfaceName)
{
	// Title, status and tooltip
	setTitle(m_UserInterfaceName + " Presets");
	setToolTip(UserInterfaceName + " presets");
	setStatusTip(UserInterfaceName + " presets");

	// Assign layout
	setLayout(&m_MainLayout);

	// Name edit
	m_PresetName.setEditable(false);
	m_PresetName.setFixedHeight(22);
	m_MainLayout.addWidget(&m_PresetName, 0, 0);

	// Save Preset
	m_SavePreset.setIcon(GetIcon("star"));
	m_SavePreset.setToolTip("Save " + m_UserInterfaceName.toLower() + " Preset");
	m_SavePreset.setStatusTip("Save " + m_UserInterfaceName.toLower() + " preset");
	m_SavePreset.setFixedWidth(22);
	m_SavePreset.setFixedHeight(22);
	m_MainLayout.addWidget(&m_SavePreset, 0, 2);

	// Rename Preset
	m_RenamePreset.setIcon(GetIcon("edit"));
	m_RenamePreset.setToolTip("Rename " + m_UserInterfaceName.toLower() + " Preset");
	m_RenamePreset.setStatusTip("Rename " + m_UserInterfaceName.toLower() + " preset");
	m_RenamePreset.setFixedWidth(22);
	m_RenamePreset.setFixedHeight(22);
	m_MainLayout.addWidget(&m_RenamePreset, 0, 3);

	// Remove preset
	m_RemovePreset.setIcon(GetIcon("cross"));
	m_RemovePreset.setToolTip("Remove " + m_UserInterfaceName.toLower() + " Preset");
	m_RemovePreset.setStatusTip("Remove " + m_UserInterfaceName.toLower() + " preset");
	m_RemovePreset.setFixedWidth(22);
	m_RemovePreset.setFixedHeight(22);
	m_MainLayout.addWidget(&m_RemovePreset, 0, 4);

	// Load presets
	m_LoadPresets.setIcon(GetIcon("folders"));
	m_LoadPresets.setToolTip("Load " + m_UserInterfaceName.toLower() + " presets from file");
	m_LoadPresets.setStatusTip("Load " + m_UserInterfaceName.toLower() + " presets from file");
	m_LoadPresets.setFixedWidth(22);
	m_LoadPresets.setFixedHeight(22);
//	m_MainLayout.addWidget(&m_LoadPresets, 0, 5);

	// Save presets
	m_SavePresets.setIcon(GetIcon("disks"));
	m_SavePresets.setToolTip("Save " + m_UserInterfaceName.toLower() + " presets to file");
	m_SavePresets.setStatusTip("Save " + m_UserInterfaceName.toLower() + " presets to file");
	m_SavePresets.setFixedWidth(22);
	m_SavePresets.setFixedHeight(22);
//	m_MainLayout.addWidget(&m_SavePresets, 0, 6);

	// Connections
	connect(&m_SavePreset, SIGNAL(clicked()), this, SLOT(OnSavePreset()));
	connect(&m_RenamePreset, SIGNAL(clicked()), this, SLOT(OnRenamePreset()));
	connect(&m_RemovePreset, SIGNAL(clicked()), this, SLOT(OnRemovePreset()));
	connect(&m_LoadPresets, SIGNAL(clicked()), this, SLOT(OnLoadPresets()));
	connect(&m_SavePresets, SIGNAL(clicked()), this, SLOT(OnSavePresets()));
	connect(&m_PresetName, SIGNAL(currentIndexChanged(int)), this, SLOT(OnCurrentIndexChanged(int)));
}

void QPresetsWidgetBase::OnLoadPreset(void)
{
	if (m_PresetName.currentText().isEmpty())
		return;

	emit LoadPreset(m_PresetName.currentText());
}

void QPresetsWidgetBase::OnLoadPreset(const QString& PresetName)
{
	if (PresetName.isEmpty())
		return;

	// Index
	const int Index = m_PresetName.findText(PresetName);

	if (Index >= 0)
	{
		// Update combo box to reflect selection
		m_PresetName.setCurrentIndex(Index);

		emit LoadPreset(PresetName);
	}
	else
	{
		emit LoadPreset("Default");
	}
}

void QPresetsWidgetBase::OnSavePreset(void)
{
	QInputDialogEx InputDialog;

	InputDialog.setTextValue(m_PresetName.currentText());
	InputDialog.setLabelText("Name");

	if (InputDialog.exec() == QDialog::Rejected)
		return;

	if (InputDialog.textValue().isEmpty())
		return;

	emit SavePreset(InputDialog.textValue());
}

void QPresetsWidgetBase::OnRenamePreset(void)
{
	if (m_PresetName.currentText().isEmpty())
		return;

	QInputDialogEx InputDialog;

	InputDialog.setTextValue(m_PresetName.currentText());
	InputDialog.setLabelText("Name");

	if (InputDialog.exec() == QDialog::Rejected)
		return;

	// Get new name
	QString Name = InputDialog.textValue();

	// Rename
	if (!Name.isEmpty())
		this->RenamePreset(m_PresetName.currentIndex(), Name);
}

void QPresetsWidgetBase::OnRemovePreset(void)
{
	this->RemovePreset();
}

void QPresetsWidgetBase::OnSavePresets(void)
{
	this->SavePresets(true);
}

void QPresetsWidgetBase::OnLoadPresets(void)
{
	this->LoadPresets(true);
}

void QPresetsWidgetBase::OnCurrentIndexChanged(int Index)
{
	LoadPreset(m_PresetName.currentText());
	m_SavePreset.setEnabled(true);
	m_RenamePreset.setEnabled(Index >= 0);
	m_RemovePreset.setEnabled(Index >= 0);
	m_LoadPresets.setEnabled(true);
	m_SavePresets.setEnabled(m_PresetName.count() > 0);
}