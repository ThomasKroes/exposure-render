
#include "StatisticsWidget.h"
#include "MainWindow.h"
#include "Statistics.h"
#include "RenderThread.h"

#define MB 1024.0f * 1024.0f

QString FormatVector(const Vec3f& Vector, const int& Precision = 2)
{
	return "[" + QString::number(Vector.x, 'f', Precision) + ", " + QString::number(Vector.y, 'f', Precision) + ", " + QString::number(Vector.z, 'f', Precision) + "]";
}

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
	m_Tree.header()->setWindowIcon(QIcon(":/Images/table-export.png"));
	
	PopulateTree();

	// Notify us when rendering begins and ends, and before/after each rendered frame
	connect(&gRenderStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gRenderStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	connect(&gRenderStatus, SIGNAL(PreRenderFrame()), this, SLOT(OnPreRenderFrame()));
	connect(&gRenderStatus, SIGNAL(PostRenderFrame()), this, SLOT(OnPostRenderFrame()));
	connect(&gRenderStatus, SIGNAL(BufferSizeChanged(const QString&, const int&)), this, SLOT(OnBufferSizeChanged(const QString&, const int&)));
}

QSize QStatisticsWidget::sizeHint() const
{
	return QSize(500, 900);
}

void QStatisticsWidget::PopulateTree(void)
{
	// Performance
	QTreeWidgetItem* pPerformance = AddItem(NULL, "Performance");
	pPerformance->setIcon(0, QIcon(":/Images/alarm-clock.png"));

	AddItem(pPerformance, "Tracer FPS", "", "Frames/Sec.");
	AddItem(pPerformance, "No. Iterations", "", "Iterations");

	// Memory
	QTreeWidgetItem* pMemory = AddItem(NULL, "Memory");
	pMemory->setIcon(0, QIcon(":/Images/memory.png"));

	// Volume
	QTreeWidgetItem* pVolume = AddItem(NULL, "Volume");
	pVolume->setIcon(0, QIcon(":/Images/grid.png"));

	AddItem(pVolume, "File", "");
	AddItem(pVolume, "Bounding Box");
	AddItem(pVolume, "Resolution");
	AddItem(pVolume, "Spacing");
	AddItem(pVolume, "Scale");
	AddItem(pVolume, "No. Voxels");
	AddItem(pVolume, "Density Range");

	// Camera
	QTreeWidgetItem* pCamera = AddItem(NULL, "Camera");
	pCamera->setIcon(0, QIcon(":/Images/camera.png"));

	AddItem(pCamera, "Resolution", "", "Pixels");
	AddItem(pCamera, "Position");
	AddItem(pCamera, "Target");
	AddItem(pCamera, "Up Vector");
	AddItem(pCamera, "Aperture Size");
	AddItem(pCamera, "Field Of View");
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
	if (!Scene())
		return;
		
	// Volume
	UpdateStatistic("File", gpRenderThread->GetFileName());
	UpdateStatistic("Bounding Box", "[" + QString::number(Scene()->m_BoundingBox.m_MinP.x) + ", " + QString::number(Scene()->m_BoundingBox.m_MinP.y) + ", " + QString::number(Scene()->m_BoundingBox.m_MinP.z) + "] - [" + QString::number(Scene()->m_BoundingBox.m_MaxP.x) + ", " + QString::number(Scene()->m_BoundingBox.m_MaxP.y) + ", " + QString::number(Scene()->m_BoundingBox.m_MaxP.z) + "]");
	UpdateStatistic("Resolution", "[" + QString::number(Scene()->m_Resolution.m_XYZ.x) + ", " + QString::number(Scene()->m_Resolution.m_XYZ.y) + ", " + QString::number(Scene()->m_Resolution.m_XYZ.z) + "]");
	UpdateStatistic("Spacing", "[" + QString::number(Scene()->m_Spacing.x) + ", " + QString::number(Scene()->m_Spacing.y) + ", " + QString::number(Scene()->m_Spacing.z) + "]");
	UpdateStatistic("Scale", "[" + QString::number(Scene()->m_Scale.x) + ", " + QString::number(Scene()->m_Scale.y) + ", " + QString::number(Scene()->m_Scale.z) + "]");
	UpdateStatistic("No. Voxels", QString::number(Scene()->m_NoVoxels));
	UpdateStatistic("Density Range", "[" + QString::number(Scene()->m_IntensityRange.m_Min) + " - " + QString::number(Scene()->m_IntensityRange.m_Max) + "]");

	// Expand all tree items
	OnExpandAll(true);
}

void QStatisticsWidget::OnRenderEnd(void)
{
	// Collapse all tree items
	OnExpandAll(false);
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

	UpdateStatistic("Tracer FPS", QString::number(Scene()->m_FPS.m_FilteredDuration, 'f', 2));
	UpdateStatistic("No. Iterations", QString::number(gpRenderThread->GetNoIterations()));

	// Camera
	UpdateStatistic("Resolution", QString::number(Scene()->m_Camera.m_Film.m_Resolution.m_XY.x) + " x " + QString::number(Scene()->m_Camera.m_Film.m_Resolution.m_XY.y));
	UpdateStatistic("Position", FormatVector(Scene()->m_Camera.m_From));
	UpdateStatistic("Target", FormatVector(Scene()->m_Camera.m_Target));
	UpdateStatistic("Up Vector", FormatVector(Scene()->m_Camera.m_Up));
	UpdateStatistic("Aperture Size", QString::number(Scene()->m_Camera.m_Aperture.m_Size, 'f', 2));
	UpdateStatistic("Field Of View", QString::number(Scene()->m_Camera.m_FovV, 'f', 2));
}

void QStatisticsWidget::OnBufferSizeChanged(const QString& Name, const int& Size)
{
	QList<QTreeWidgetItem*> Items = m_Tree.findItems(Name, Qt::MatchFlag::MatchRecursive, 0);

	QTreeWidgetItem* pItem = FindItem(Name);

	if (!pItem)
		return;

	const QString SizeString = QString::number(Size / powf(1024.0f, 2.0f), 'f', 2);

	if (Items.size() <= 0)
	{
		AddItem(pItem, Name, SizeString, "");
	}
	else
	{
		UpdateStatistic(Name, SizeString);
	}
}

void QStatisticsWidget::OnExpandAll(const bool& Expand)
{
	QList<QTreeWidgetItem*> Items = m_Tree.findItems("*", Qt::MatchFlag::MatchRecursive | Qt::MatchFlag::MatchWildcard, 0);

	foreach (QTreeWidgetItem* pItem, Items)
		pItem->setExpanded(Expand);
}

QTreeWidgetItem* QStatisticsWidget::FindItem(const QString& Name)
{
	QList<QTreeWidgetItem*> Items = m_Tree.findItems(Name, Qt::MatchFlag::MatchRecursive, 0);

	if (Items.size() <= 0)
		return NULL;
	else
		return Items[0];
}
