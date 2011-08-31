
#include "StatisticsWidget.h"
#include "MainWindow.h"
#include "Statistics.h"
#include "Scene.h"

#define MB 1024.0f * 1024.0f

QStatisticsWidget::QStatisticsWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_Group(),
	m_Tree()
{
	// Set the size policy, making sure the widget fits nicely in the layout
	setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

	// Status and tooltip
	setToolTip("Statistics");
	setStatusTip("Statistics");

	// Set the main layout
	setLayout(&m_MainLayout);

	// Add the group
	m_MainLayout.addWidget(&m_Group);

	// Set the layout
	m_Group.setLayout(&m_GroupLayout);

	// Add the tree widget
	m_GroupLayout.addWidget(&m_Tree);

	// Configure tree
	m_Tree.setColumnCount(3);

	QStringList ColumnNames;

	ColumnNames << "Property" << "Value" << "Unit";

	m_Tree.setHeaderLabels(ColumnNames);

	// Configure headers
	m_Tree.header()->setResizeMode(0, QHeaderView::ResizeMode::ResizeToContents);
//	m_Tree.header()->setResizeMode(1, QHeaderView::ResizeMode::ResizeToContents);
//	m_Tree.header()->setResizeMode(2, QHeaderView::ResizeMode::ResizeToContents);

	PopulateTree();
}

void QStatisticsWidget::PopulateTree(void)
{
	// Performance
	QTreeWidgetItem* pPerformance = AddItem(NULL, "Performance");

	AddItem(pPerformance, "Tracer FPS", "", "Frames/Sec.");
	AddItem(pPerformance, "No. Iterations", "", "Iterations");

	// Memory
	QTreeWidgetItem* pMemory = AddItem(NULL, "Memory");

	AddItem(pMemory, "Volume (Cuda)", "", "MB");
	AddItem(pMemory, "HDR Accumulation Buffer (Cuda)", "", "MB");
	AddItem(pMemory, "HDR Frame Buffer (Cuda)", "", "MB");
	AddItem(pMemory, "HDR Frame Buffer Blur (Cuda)", "", "MB");
	AddItem(pMemory, "LDR Estimation Buffer (Cuda)", "", "MB");

	// Volume
	QTreeWidgetItem* pVolume = AddItem(NULL, "Volume");

	AddItem(pVolume, "File", "");
	AddItem(pVolume, "Bounding Box");
	AddItem(pVolume, "Resolution");
	AddItem(pVolume, "Spacing");
	AddItem(pVolume, "Scale");
	AddItem(pVolume, "No. Voxels");
	AddItem(pVolume, "Density Range");
}

QTreeWidgetItem* QStatisticsWidget::AddItem(QTreeWidgetItem* pParent, const QString& Property, const QString& Value, const QString& Unit)
{
	// Create new item
	QTreeWidgetItem* pItem = new QTreeWidgetItem(pParent);
	pItem->setText(0, Property);
	pItem->setText(1, Value);
	pItem->setText(2, Unit);

	if (!pParent)
		m_Tree.addTopLevelItem(pItem);

	return pItem;
}

void QStatisticsWidget::UpdateStatistic(const QString& Property, const QString& Value)
{
	QList<QTreeWidgetItem*> Items = m_Tree.findItems(Property, Qt::MatchFlag::MatchRecursive, 0);

	foreach (QTreeWidgetItem* pItem, Items)
	{
		pItem->setText(1, Value);
	}
}

void QStatisticsWidget::OnRenderBegin(void)
{
	// Create the necessary connections so that we can present up to date statistics
	connect(gpRenderThread, SIGNAL(MemoryAllocate()), this, SLOT(OnMemoryAllocate()));
	connect(gpRenderThread, SIGNAL(MemoryFree()), this, SLOT(OnMemoryFree()));
	connect(gpRenderThread, SIGNAL(PreFrame()), this, SLOT(OnPreFrame()));
	connect(gpRenderThread, SIGNAL(PostFrame()), this, SLOT(OnPostFrame()));

	// Memory
	UpdateStatistic("Volume (Cuda)", QString::number((float)gpRenderThread->m_SizeVolume / powf(1024.0f, 2.0f), 'f', 2));
	UpdateStatistic("HDR Accumulation Buffer (Cuda)", QString::number((float)gpRenderThread->m_SizeHdrAccumulationBuffer / powf(1024.0f, 2.0f), 'f', 2));
	UpdateStatistic("HDR Frame Buffer (Cuda)", QString::number((float)gpRenderThread->m_SizeHdrFrameBuffer / powf(1024.0f, 2.0f), 'f', 2));
	UpdateStatistic("HDR Frame Buffer Blur (Cuda)", QString::number((float)gpRenderThread->m_SizeHdrBlurFrameBuffer / powf(1024.0f, 2.0f), 'f', 2));
	UpdateStatistic("LDR Estimation Buffer (Cuda)", QString::number((float)gpRenderThread->m_SizeLdrFrameBuffer / powf(1024.0f, 2.0f), 'f', 2));

	// Volume
	UpdateStatistic("File", gpRenderThread->m_FileName);
	UpdateStatistic("Bounding Box", "[" + QString::number(gpScene->m_BoundingBox.m_MinP.x) + ", " + QString::number(gpScene->m_BoundingBox.m_MinP.y) + ", " + QString::number(gpScene->m_BoundingBox.m_MinP.z) + "] - [" + QString::number(gpScene->m_BoundingBox.m_MaxP.x) + ", " + QString::number(gpScene->m_BoundingBox.m_MaxP.y) + ", " + QString::number(gpScene->m_BoundingBox.m_MaxP.z) + "]");
	UpdateStatistic("Resolution", "[" + QString::number(gpScene->m_Resolution.m_XYZ.x) + ", " + QString::number(gpScene->m_Resolution.m_XYZ.y) + ", " + QString::number(gpScene->m_Resolution.m_XYZ.z) + "]");
	UpdateStatistic("Spacing", "[" + QString::number(gpScene->m_Spacing.x) + ", " + QString::number(gpScene->m_Spacing.y) + ", " + QString::number(gpScene->m_Spacing.z) + "]");
	UpdateStatistic("Scale", "[" + QString::number(gpScene->m_Scale.x) + ", " + QString::number(gpScene->m_Scale.y) + ", " + QString::number(gpScene->m_Scale.z) + "]");
	UpdateStatistic("No. Voxels", QString::number(gpScene->m_NoVoxels));
	UpdateStatistic("Density Range", "[" + QString::number(gpScene->m_IntensityRange.m_Min) + " - " + QString::number(gpScene->m_IntensityRange.m_Max) + "]");
}

void QStatisticsWidget::OnRenderEnd(void)
{

}

void QStatisticsWidget::OnMemoryAllocate(void)
{

}

void QStatisticsWidget::OnMemoryFree(void)
{

}

void QStatisticsWidget::OnPreFrame(void)
{

}


void QStatisticsWidget::OnPostFrame(void)
{
	UpdateStatistic("Tracer FPS", QString::number(gpScene->m_FPS.m_FilteredDuration, 'f'));
	UpdateStatistic("No. Iterations", QString::number(gpRenderThread->m_N));
}