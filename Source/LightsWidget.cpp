/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "LightsWidget.h"

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
//	m_LightList.setSortingEnabled(true);
	m_MainLayout.addWidget(&m_LightList, 0, 0, 1, 6);

	// Add light
	m_AddLight.setIcon(GetIcon("light-bulb--plus"));
	m_AddLight.setToolTip("Add light");
	m_AddLight.setStatusTip("Add a new light to the scene");
 	m_AddLight.setFixedWidth(24);
	m_AddLight.setFixedHeight(24);
	m_MainLayout.addWidget(&m_AddLight, 1, 0);

	// Remove light
	m_RemoveLight.setIcon(GetIcon("light-bulb--minus"));
	m_RemoveLight.setToolTip("Remove light");
	m_RemoveLight.setStatusTip("Remove the selected light from the scene");
	m_RemoveLight.setFixedWidth(24);
	m_RemoveLight.setFixedHeight(24);
	m_MainLayout.addWidget(&m_RemoveLight, 1, 1);

	// Rename light
	m_RenameLight.setIcon(GetIcon("light-bulb--pencil"));
	m_RenameLight.setToolTip("Rename light");
	m_RenameLight.setStatusTip("Rename the selected light");
 	m_RenameLight.setFixedWidth(24);
	m_RenameLight.setFixedHeight(24);
	m_MainLayout.addWidget(&m_RenameLight, 1, 2);

 	connect(&m_LightList, SIGNAL(itemSelectionChanged()), this, SLOT(OnLightSelectionChanged()));
	connect(&gLighting, SIGNAL(LightSelectionChanged(QLight*)), this, SLOT(OnLightSelectionChanged(QLight*)));
 	connect(&m_AddLight, SIGNAL(clicked()), this, SLOT(OnAddLight()));
 	connect(&m_RemoveLight, SIGNAL(clicked()), this, SLOT(OnRemoveLight()));
	connect(&m_RenameLight, SIGNAL(clicked()), this, SLOT(OnRenameLight()));
 	connect(&gLighting, SIGNAL(Changed()), this, SLOT(UpdateLightList()));

	OnLightSelectionChanged();
}

void QLightsWidget::UpdateLightList(void)
{
	m_LightList.clear();

	m_LightList.setFocus();

	for (int i = 0; i < gLighting.m_Lights.size(); i++)
	{
		// Create new list item
		QLightItem* pLightItem = new QLightItem(&m_LightList, &gLighting.m_Lights[i]);

		// Add the item
		m_LightList.addItem(pLightItem);
	}

	m_RemoveLight.setEnabled(m_LightList.currentRow() >= 0);
	m_RenameLight.setEnabled(m_LightList.currentRow() >= 0);
	m_CopyLight.setEnabled(m_LightList.currentRow() >= 0);

	for (int i = 0; i < m_LightList.count(); i++)
	{
		if (((QLightItem*)m_LightList.item(i))->m_pLight == gLighting.GetSelectedLight())
		{
			m_LightList.blockSignals(true);

			m_LightList.setCurrentRow(i, QItemSelectionModel::Select);
			m_LightList.setFocus();

			m_LightList.blockSignals(false);
		}
	}
}

void QLightsWidget::OnLightSelectionChanged(void)
{
	if (m_LightList.currentRow() < 0)
		return;

	gLighting.SetSelectedLight(m_LightList.currentRow());

	m_RemoveLight.setEnabled(m_LightList.currentRow() >= 0);
	m_RenameLight.setEnabled(m_LightList.currentRow() >= 0);
	m_CopyLight.setEnabled(m_LightList.currentRow() >= 0);
}

void QLightsWidget::OnLightSelectionChanged(QLight* pLight)
{
	for (int i = 0; i < m_LightList.count(); i++)
	{
		if (((QLightItem*)m_LightList.item(i))->m_pLight == pLight)
		{
			m_LightList.blockSignals(true);

			m_LightList.setCurrentRow(i, QItemSelectionModel::Select);
			m_LightList.setFocus();

			m_LightList.blockSignals(false);
		}
	}
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
	if (m_LightList.currentRow() < 0)
		return;

	gLighting.RemoveLight(m_LightList.currentRow());
}

void QLightsWidget::OnRenameLight(void)
{
	if (m_LightList.currentRow() < 0)
		return;

	QInputDialogEx InputDialog;

	InputDialog.setTextValue(gLighting.m_Lights[m_LightList.currentRow()].GetName());
	InputDialog.setWindowTitle("Choose name for light");
	InputDialog.setLabelText("Light Name");
	InputDialog.setOkButtonText("Rename");
	
	InputDialog.move(m_RenameLight.rect().center());
	
	InputDialog.exec();

	if (InputDialog.textValue().isEmpty())
		return;

	gLighting.RenameLight(m_LightList.currentRow(), InputDialog.textValue());
}