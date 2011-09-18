
// Precompiled headers
#include "Stable.h"

#include "StatisticsWidget.h"
#include "MainWindow.h"
#include "Statistics.h"
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
	connect(&gRenderStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gRenderStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	connect(&gRenderStatus, SIGNAL(PreRenderFrame()), this, SLOT(OnPreRenderFrame()));
	connect(&gRenderStatus, SIGNAL(PostRenderFrame()), this, SLOT(OnPostRenderFrame()));
	connect(&gRenderStatus, SIGNAL(StatisticChanged(const QString&, const QString&, const QString&, const QString&, const QString&)), this, SLOT(OnStatisticChanged(const QString&, const QString&, const QString&, const QString&, const QString&)));
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

void QStatisticsWidget::OnPreRenderFrame(void)
{
	if (!Scene())
		return;
}

void QStatisticsWidget::OnPostRenderFrame(void)
{
	if (!Scene())
		return;
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
