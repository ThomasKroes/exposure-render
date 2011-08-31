
#include "LightingPresetsWidget.h"
#include "RenderThread.h"

QLightingPresetsWidget::QLightingPresetsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_PresetList(),
	m_PresetName(),
	m_LoadPreset(),
	m_SavePreset(),
	m_RemovePreset(),
	m_LoadPresets(),
	m_SavePresets()
{
	// Title, status and tooltip
	setTitle("Presets");
	setToolTip("Lighting Presets");
	setStatusTip("Lighting Presets");

	// Apply main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	// Preset list
//	m_PresetList.setParent(this);
	m_PresetList.setSelectionMode(QAbstractItemView::SingleSelection);
	m_PresetList.setAlternatingRowColors(true);
	m_PresetList.setSortingEnabled(true);
	m_MainLayout.addWidget(&m_PresetList, 1, 0, 1, 6);

	// Preset name
	m_MainLayout.addWidget(&m_PresetName, 0, 0);

	// Save preset
	m_SavePreset.setParent(this);
	m_SavePreset.setEnabled(false);
	m_SavePreset.setText("Save");
	m_SavePreset.setToolTip("Save Preset");
	m_SavePreset.setStatusTip("Save the lighting preset");
	m_SavePreset.setFixedWidth(20);
	m_SavePreset.setFixedHeight(20);
	m_MainLayout.addWidget(&m_SavePreset, 0, 1);

	// Load preset
	m_LoadPreset.setParent(this);
	m_LoadPreset.setEnabled(false);
	m_LoadPreset.setText("Load");
	m_LoadPreset.setToolTip("Load Preset");
	m_LoadPreset.setStatusTip("Load the selected lighting preset");
	m_LoadPreset.setFixedWidth(20);
	m_LoadPreset.setFixedHeight(20);
	m_MainLayout.addWidget(&m_LoadPreset, 0, 2);

	// Remove preset
	m_RemovePreset.setParent(this);
	m_RemovePreset.setEnabled(false);
	m_RemovePreset.setText("Remove");
	m_RemovePreset.setToolTip("Remove Preset");
	m_RemovePreset.setStatusTip("Remove the selected lighting preset");
	m_RemovePreset.setFixedWidth(20);
	m_RemovePreset.setFixedHeight(20);
	m_MainLayout.addWidget(&m_RemovePreset, 0, 3);

	// Load presets
	m_LoadPresets.setParent(this);
	m_LoadPresets.setEnabled(false);
	m_LoadPresets.setText("Load");
	m_LoadPresets.setToolTip("Load Presets");
	m_LoadPresets.setStatusTip("Load lighting presets from file");
	m_LoadPresets.setFixedWidth(20);
	m_LoadPresets.setFixedHeight(20);
	m_MainLayout.addWidget(&m_LoadPresets, 0, 4);

	// Save presets
	m_SavePresets.setParent(this);
	m_SavePresets.setEnabled(false);
	m_SavePresets.setText("Save");
	m_SavePresets.setToolTip("Save Presets");
	m_SavePresets.setStatusTip("Save lighting presets from file");
	m_SavePresets.setFixedWidth(20);
	m_SavePresets.setFixedHeight(20);
	m_MainLayout.addWidget(&m_SavePresets, 0, 5);
}