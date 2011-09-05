
#include "LightsWidget.h"

QLightsWidget::QLightsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_LightList(),
	m_LightName(),
	m_AddLight(),
	m_RemoveLight(),
	m_RenameLight()
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
	m_MainLayout.addWidget(&m_LightList, 0, 0, 1, 4);

	// Light name
	m_LightName.setFixedHeight(22);
	m_MainLayout.addWidget(&m_LightName, 1, 0);

	// Add light
	m_AddLight.setIcon(QIcon(":/Images/light-bulb--plus.png"));
	m_AddLight.setToolTip("Add light");
	m_AddLight.setStatusTip("Add a new light to the scene");
	m_AddLight.setFixedWidth(22);
	m_AddLight.setFixedHeight(22);
	m_MainLayout.addWidget(&m_AddLight, 1, 1);

	// Remove light
	m_RemoveLight.setIcon(QIcon(":/Images/light-bulb--minus.png"));
	m_RemoveLight.setToolTip("Remove light");
	m_RemoveLight.setStatusTip("Remove a light from the scene");
	m_RemoveLight.setFixedWidth(22);
	m_RemoveLight.setFixedHeight(22);
	m_MainLayout.addWidget(&m_RemoveLight, 1, 2);

	// Rename light
	m_RenameLight.setIcon(QIcon(":/Images/light-bulb--pencil.png"));
	m_RenameLight.setToolTip("Rename light");
	m_RenameLight.setStatusTip("Rename a light from the scene");
	m_RenameLight.setFixedWidth(22);
	m_RenameLight.setFixedHeight(22);
	m_MainLayout.addWidget(&m_RenameLight, 1, 3);

 	connect(&m_LightList, SIGNAL(itemSelectionChanged()), this, SLOT(OnLightSelectionChanged()));
 	connect(&m_AddLight, SIGNAL(clicked()), this, SLOT(OnAddLight()));
 	connect(&m_RemoveLight, SIGNAL(clicked()), this, SLOT(OnRemoveLight()));
 	connect(&m_LightName, SIGNAL(textChanged(const QString&)), this, SLOT(OnPresetNameChanged(const QString&)));
 	connect(&m_LightList, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(OnLightItemChanged(QListWidgetItem*)));
	connect(&gLighting, SIGNAL(LightingChanged()), this, SLOT(UpdateLightList()));
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

	// Get current row
	const int CurrentRow = m_LightList.currentRow();

	m_AddLight.setEnabled(m_LightName.text().isEmpty() ? false : true);
	m_RemoveLight.setEnabled(CurrentRow >= 0);
}

void QLightsWidget::OnLightSelectionChanged(void)
{
	// Get current row
	int CurrentRow = m_LightList.currentRow();

	m_RemoveLight.setEnabled(CurrentRow >= 0);

	if (CurrentRow < 0)
		return;

	m_LightName.setText(m_LightList.currentItem()->text());

	QLightItem* pLightItem = dynamic_cast<QLightItem*>(m_LightList.currentItem());

	if (pLightItem)
		gLighting.SetSelectedLight(pLightItem->m_pLight);
}

void QLightsWidget::OnLightItemChanged(QListWidgetItem* pWidgetItem)
{
	QLightItem* pLightItem = dynamic_cast<QLightItem*>(m_LightList.currentItem());

	if (pLightItem)
		pLightItem->m_pLight->SetName(pWidgetItem->text());
}

void QLightsWidget::OnPresetNameChanged(const QString& Text)
{
	m_AddLight.setEnabled(Text.length() > 0);
}

void QLightsWidget::OnAddLight(void)
{
	if (m_LightName.text().isEmpty())
		return;

	QLight NewLight;
	NewLight.SetName(m_LightName.text());

	gLighting.AddLight(NewLight);

	UpdateLightList();
}

void QLightsWidget::OnRemoveLight(void)
{
	// Get current row
	const int CurrentRow = m_LightList.currentRow();

	if (CurrentRow < 0)
		return;

	gLighting.m_Lights.removeAt(CurrentRow);

	UpdateLightList();
}