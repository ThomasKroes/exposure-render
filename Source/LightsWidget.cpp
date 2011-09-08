
#include "LightsWidget.h"
#include "Controls.h"

QLightsWidget::QLightsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_LightList(),
	m_AddLight(),
	m_RemoveLight(),
	m_RenameLight(),
	m_CopyLight()
{
	// Title, status and tooltip
	setTitle("Lights");
	setToolTip("Lights");
	setStatusTip("Lights");

	// Apply main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	// Lights list
	m_LightList.setSelectionMode(QAbstractItemView::SingleSelection);
	m_LightList.setAlternatingRowColors(true);
	m_LightList.setSortingEnabled(true);
	m_MainLayout.addWidget(&m_LightList, 0, 0, 1, 6);

	// Add light
// 	m_AddLight.setText("Add");
	m_AddLight.setIcon(QIcon(":/Images/light-bulb--plus.png"));
	m_AddLight.setToolTip("Add light");
	m_AddLight.setStatusTip("Add a new light to the scene");
 	m_AddLight.setFixedWidth(24);
	m_AddLight.setFixedHeight(24);
	m_MainLayout.addWidget(&m_AddLight, 1, 0);

	// Remove light
// 	m_RemoveLight.setText("Remove");
	m_RemoveLight.setIcon(QIcon(":/Images/light-bulb--minus.png"));
	m_RemoveLight.setToolTip("Remove light");
	m_RemoveLight.setStatusTip("Remove the selected light from the scene");
	m_RemoveLight.setFixedWidth(24);
	m_RemoveLight.setFixedHeight(24);
	m_MainLayout.addWidget(&m_RemoveLight, 1, 1);

	// Rename light
// 	m_RenameLight.setText("Rename");
	m_RenameLight.setIcon(QIcon(":/Images/light-bulb--pencil.png"));
	m_RenameLight.setToolTip("Rename light");
	m_RenameLight.setStatusTip("Rename the selected light");
 	m_RenameLight.setFixedWidth(24);
	m_RenameLight.setFixedHeight(24);
	m_MainLayout.addWidget(&m_RenameLight, 1, 2);

	// Copy light
// 	m_CopyLight.setText("Copy");
	m_CopyLight.setIcon(QIcon(":/Images/document-copy.png"));
	m_CopyLight.setToolTip("Copy light");
	m_CopyLight.setStatusTip("Copy the selected light");
 	m_CopyLight.setFixedWidth(24);
	m_CopyLight.setFixedHeight(24);
	m_MainLayout.addWidget(&m_CopyLight, 1, 3);

	// Reset
	m_Reset.setIcon(QIcon(":/Images/document-copy.png"));
	m_Reset.setToolTip("Reset Lighting");
	m_Reset.setStatusTip("Reset lighting to defaults");
	m_Reset.setFixedWidth(24);
	m_Reset.setFixedHeight(24);
	m_MainLayout.addWidget(&m_Reset, 1, 4);
	
	// Inform us when the light selection changes, a light is added/removed/renamed/copied
 	connect(&m_LightList, SIGNAL(itemSelectionChanged()), this, SLOT(OnLightSelectionChanged()));
	connect(&gLighting, SIGNAL(LightSelectionChanged(QLight*, QLight*)), this, SLOT(OnLightSelectionChanged(QLight*, QLight*)));
 	connect(&m_AddLight, SIGNAL(clicked()), this, SLOT(OnAddLight()));
 	connect(&m_RemoveLight, SIGNAL(clicked()), this, SLOT(OnRemoveLight()));
	connect(&m_RenameLight, SIGNAL(clicked()), this, SLOT(OnRenameLight()));
	connect(&m_CopyLight, SIGNAL(clicked()), this, SLOT(OnCopyLight()));
 	connect(&m_LightList, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(OnLightItemChanged(QListWidgetItem*)));
	connect(&gLighting, SIGNAL(LightingChanged()), this, SLOT(UpdateLightList()));

	OnLightSelectionChanged();
}

void QLightsWidget::UpdateLightList(void)
{
	m_LightList.clear();

	for (int i = 0; i < gLighting.m_Lights.size(); i++)
	{
		// Create new list item
		QLightItem* pLightItem = new QLightItem(&m_LightList, &gLighting.m_Lights[i]);

		pLightItem->setFlags(pLightItem->flags() | Qt::ItemIsEditable);

		// Add the item
		m_LightList.addItem(pLightItem);
	}

	// Select
	if (gLighting.GetSelectedLight())
	{
		const int Index = gLighting.m_Lights.indexOf(*gLighting.GetSelectedLight());
		m_LightList.setCurrentRow(Index, QItemSelectionModel::Select);
		m_LightList.setFocus();
	}

	// Get current row
	const int CurrentRow = m_LightList.currentRow();

	m_RemoveLight.setEnabled(CurrentRow >= 0);
	m_RenameLight.setEnabled(CurrentRow >= 0);
	m_CopyLight.setEnabled(CurrentRow >= 0);
}

void QLightsWidget::OnLightSelectionChanged(void)
{
	// Get current row
	int CurrentRow = m_LightList.currentRow();

	if (CurrentRow < 0)
		return;

//	gLighting.SetSelectedLight(CurrentRow);
}

void QLightsWidget::OnLightSelectionChanged(QLight* pOldLight, QLight* pNewLight)
{
//	if (!pNewLight)
//		return;
	
//	m_LightList.update();
}

void QLightsWidget::OnLightItemChanged(QListWidgetItem* pWidgetItem)
{
	QLightItem* pLightItem = dynamic_cast<QLightItem*>(m_LightList.currentItem());

	if (pLightItem)
		pLightItem->m_pLight->SetName(pWidgetItem->text());
}

void QLightsWidget::OnAddLight(void)
{
	QInputDialogEx InputDialog;

	InputDialog.setTextValue("Light " + QString::number(gLighting.m_Lights.size() + 1));
	InputDialog.setWindowTitle("Choose name for light");
	InputDialog.setLabelText("Light Name");
	InputDialog.setOkButtonText("Add");

	InputDialog.exec();

	if (InputDialog.textValue().isEmpty())
		return;

	QLight NewLight;
	NewLight.SetName(InputDialog.textValue());

	// Add the light
	gLighting.AddLight(NewLight);
}

void QLightsWidget::OnRemoveLight(void)
{
	gLighting.RemoveLight(m_LightList.currentRow());
}

void QLightsWidget::OnRenameLight(void)
{
	// Get current row
	const int CurrentRow = m_LightList.currentRow();

	if (CurrentRow < 0)
		return;

	QInputDialogEx InputDialog;

	InputDialog.setTextValue(gLighting.m_Lights[CurrentRow].GetName());
	InputDialog.setWindowTitle("Choose name for light");
	InputDialog.setLabelText("Light Name");
	InputDialog.setOkButtonText("Rename");

	InputDialog.exec();

	if (InputDialog.textValue().isEmpty())
		return;

	gLighting.RenameLight(CurrentRow, InputDialog.textValue());
}

void QLightsWidget::OnCopyLight(void)
{
	gLighting.CopySelectedLight();
}

void QLightsWidget::OnReset(void)
{
}