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

#include "StatisticsWidget.h"
#include "MainWindow.h"
#include "RenderThread.h"

QStatisticsWidget::QStatisticsWidget(QWidget* pParent) :
	QTreeWidget(pParent),
	m_MainLayout()
{
	// Set the size policy, making sure the widget fits nicely in the layout
	setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

	// Status and tooltip
	setToolTip("Statistics");
	setStatusTip("Statistics");

	// Configure tree
	setColumnCount(3);

	QStringList ColumnNames;

	ColumnNames << "Property" << "Value" << "Unit";

	setHeaderLabels(ColumnNames);

	// Configure headers
//	header()->setResizeMode(0, QHeaderView::ResizeToContents);
//	header()->setResizeMode(1, QHeaderView::ResizeToContents);
//	header()->setResizeMode(2, QHeaderView::ResizeToContents);
	header()->resizeSection(0, 260);
	header()->resizeSection(1, 150);
	header()->resizeSection(2, 100);
	header()->setWindowIcon(GetIcon("table-export"));
	header()->setVisible(false);

	PopulateTree();
	
	// Notify us when rendering begins and ends, and before/after each rendered frame
	connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	connect(&gStatus, SIGNAL(StatisticChanged(const QString&, const QString&, const QString&, const QString&, const QString&)), this, SLOT(OnStatisticChanged(const QString&, const QString&, const QString&, const QString&, const QString&)));
}

QSize QStatisticsWidget::sizeHint() const
{
	return QSize(550, 900);
}

void QStatisticsWidget::PopulateTree(void)
{
	// Populate tree with top-level items
	AddItem(NULL, "Performance", "", "", "application-monitor");
	AddItem(NULL, "Volume", "", "", "grid");
	AddItem(NULL, "Memory", "", "", "memory");
	AddItem(NULL, "Camera", "", "", "camera");
	AddItem(NULL, "Graphics Card", "", "", "graphic-card");
}

QTreeWidgetItem* QStatisticsWidget::AddItem(QTreeWidgetItem* pParent, const QString& Property, const QString& Value, const QString& Unit, const QString& Icon)
{
	// Create new item
	QTreeWidgetItem* pItem = new QTreeWidgetItem(pParent);

	// Set item properties
	pItem->setText(0, Property);
	pItem->setText(1, Value);
	pItem->setText(2, Unit);
	pItem->setIcon(0, GetIcon(Icon));

	if (!pParent)
		addTopLevelItem(pItem);

	return pItem;
}

void QStatisticsWidget::UpdateStatistic(const QString& Group, const QString& Name, const QString& Value, const QString& Unit, const QString& Icon)
{
	QTreeWidgetItem* pGroup = FindItem(Group);

	if (!pGroup)
	{
		pGroup = AddItem(NULL, Group);
		
		AddItem(pGroup, Name, Value, Unit, Icon);
	}
	else
	{
		bool Found = false;

		for (int i = 0; i < pGroup->childCount(); i++)
		{
			if (pGroup->child(i)->text(0) == Name)
			{
				pGroup->child(i)->setText(1, Value);
				pGroup->child(i)->setText(2, Unit);

				Found = true;
			}
		}

		if (!Found)
			AddItem(pGroup, Name, Value, Unit, Icon);
	}
}

void QStatisticsWidget::OnRenderBegin(void)
{
	// Expand all tree items
	ExpandAll(true);
}

void QStatisticsWidget::OnRenderEnd(void)
{
	// Collapse all tree items
	ExpandAll(false);

	// Remove 2nd order children
	RemoveChildren("Performance");
	RemoveChildren("Volume");
	RemoveChildren("Memory");
	RemoveChildren("Camera");
	RemoveChildren("Graphics Card");
}

void QStatisticsWidget::OnStatisticChanged(const QString& Group, const QString& Name, const QString& Value, const QString& Unit /*= ""*/, const QString& Icon /*= ""*/)
{
	UpdateStatistic(Group, Name, Value, Unit, Icon);
}

void QStatisticsWidget::ExpandAll(const bool& Expand)
{
	QList<QTreeWidgetItem*> Items = findItems("*", Qt::MatchRecursive | Qt::MatchWildcard, 0);

	foreach (QTreeWidgetItem* pItem, Items)
		pItem->setExpanded(Expand);
}

QTreeWidgetItem* QStatisticsWidget::FindItem(const QString& Name)
{
	QList<QTreeWidgetItem*> Items = findItems(Name, Qt::MatchRecursive, 0);

	if (Items.size() <= 0)
		return NULL;
	else
		return Items[0];
}

void QStatisticsWidget::RemoveChildren(const QString& Name)
{
	QTreeWidgetItem* pItem = FindItem(Name);

	if (pItem)
		qDeleteAll(pItem->takeChildren());
}
