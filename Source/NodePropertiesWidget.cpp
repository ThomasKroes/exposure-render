
#include "NodePropertiesWidget.h"
#include "TransferFunction.h"
#include "NodeItem.h"

QNodePropertiesWidget::QNodePropertiesWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_SelectionLabel(),
	m_SelectionLayout(),
	m_NodeSelection(),
	m_PreviousNode(),
	m_NextNode(),
	m_DeleteNode(),
	m_IntensityLabel(),
	m_IntensitySlider(),
	m_IntensitySpinBox(),
	m_OpacityLabel(),
	m_OpacitySlider(),
	m_OpacitySpinBox(),
	m_KdColor(),
	m_KsColor(),
	m_KtColor(),
	m_RoughnessLabel(),
	m_RoughnessSlider(),
	m_RoughnessSpinBox()
{
	// Title, status and tooltip
	setTitle("Node Properties");
	setToolTip("Node Properties");
	setStatusTip("Node Properties");

	// Main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
//	m_pMainLayout.setContentsMargins(0, 0, 0, 0);

	setLayout(&m_MainLayout);

	setAlignment(Qt::AlignTop);

	// Selection
	m_SelectionLabel.setText("Selection");
	m_SelectionLabel.setStatusTip("Node selection");
	m_SelectionLabel.setToolTip("Node selection");
	m_MainLayout.addWidget(&m_SelectionLabel, 0, 0);

	m_SelectionLayout.setAlignment(Qt::AlignTop);
	m_SelectionLayout.setContentsMargins(0, 0, 0, 0);
	
	m_MainLayout.addLayout(&m_SelectionLayout, 0, 1, 1, 2);

	m_NodeSelection.setStatusTip("Node selection");
	m_NodeSelection.setToolTip("Node selection");
	m_SelectionLayout.addWidget(&m_NodeSelection, 0, 0);

	connect(&m_NodeSelection, SIGNAL(currentIndexChanged(int)), this, SLOT(OnNodeSelectionChanged(int)));

	// Previous node
	m_PreviousNode.setIcon(QIcon(":/Images/arrow-180.png"));
	m_PreviousNode.setStatusTip("Select previous node");
	m_PreviousNode.setToolTip("Select previous node");
	m_PreviousNode.setFixedWidth(22);
	m_PreviousNode.setFixedHeight(22);
	m_PreviousNode.updateGeometry();
	m_SelectionLayout.addWidget(&m_PreviousNode, 0, 1);

	connect(&m_PreviousNode, SIGNAL(pressed()), this, SLOT(OnPreviousNode()));

	// Next node
	m_NextNode.setIcon(QIcon(":/Images/arrow.png"));
	m_NextNode.setStatusTip("Select next node");
	m_NextNode.setToolTip("Select next node");
	m_NextNode.setFixedWidth(20);
	m_NextNode.setFixedHeight(20);
	m_SelectionLayout.addWidget(&m_NextNode, 0, 2);
	
	connect(&m_NextNode, SIGNAL(pressed()), this, SLOT(OnNextNode()));

	// Delete node
	m_DeleteNode.setIcon(QIcon(":/Images/bin.png"));
	m_DeleteNode.setStatusTip("Delete selected node");
	m_DeleteNode.setToolTip("Delete selected node");
	m_DeleteNode.setFixedWidth(20);
	m_DeleteNode.setFixedHeight(20);
	m_SelectionLayout.addWidget(&m_DeleteNode, 0, 3);
	
	connect(&m_DeleteNode, SIGNAL(pressed()), this, SLOT(OnDeleteNode()));

	// Position
	m_IntensityLabel.setText("Position");
	m_IntensityLabel.setStatusTip("Node position");
	m_IntensityLabel.setToolTip("Node position");
	m_MainLayout.addWidget(&m_IntensityLabel, 1, 0);

	m_IntensitySlider.setOrientation(Qt::Orientation::Horizontal);
	m_IntensitySlider.setStatusTip("Node position");
	m_IntensitySlider.setToolTip("Drag to change node position");
	m_IntensitySlider.setRange(0, 100);
	m_IntensitySlider.setSingleStep(1);
	m_MainLayout.addWidget(&m_IntensitySlider, 1, 1);
	
	m_IntensitySpinBox.setStatusTip("Node Position");
	m_IntensitySpinBox.setToolTip("Node Position");
    m_IntensitySpinBox.setRange(0, 100);
	m_IntensitySpinBox.setSingleStep(1);
	m_MainLayout.addWidget(&m_IntensitySpinBox, 1, 2);

	// Opacity
	m_OpacityLabel.setText("Opacity");
	m_MainLayout.addWidget(&m_OpacityLabel, 2, 0);

	m_OpacitySlider.setOrientation(Qt::Orientation::Horizontal);
	m_OpacitySlider.setStatusTip("Node Opacity");
	m_OpacitySlider.setToolTip("Node Opacity");
	m_OpacitySlider.setRange(0, 100);
	m_OpacitySlider.setSingleStep(1);
	m_MainLayout.addWidget(&m_OpacitySlider, 2, 1);
	
	m_OpacitySpinBox.setStatusTip("Node Opacity");
	m_OpacitySpinBox.setToolTip("Node Opacity");
    m_OpacitySpinBox.setRange(0, 100);
	m_OpacitySpinBox.setSingleStep(1);
	m_MainLayout.addWidget(&m_OpacitySpinBox, 2, 2);
	
	// Kd
	m_MainLayout.addWidget(new QLabel("Kd"), 3, 0);
	m_MainLayout.addWidget(&m_KdColor, 3, 1);

	// Ks
	m_MainLayout.addWidget(new QLabel("Ks"), 4, 0);
	m_MainLayout.addWidget(&m_KsColor, 4, 1);

	// Kt
	m_MainLayout.addWidget(new QLabel("Kt"), 5, 0);
	m_MainLayout.addWidget(&m_KtColor, 5, 1);

	// Setup connections for position
	connect(&m_IntensitySlider, SIGNAL(valueChanged(int)), &m_IntensitySpinBox, SLOT(setValue(int)));
	connect(&m_IntensitySpinBox, SIGNAL(valueChanged(int)), &m_IntensitySlider, SLOT(setValue(int)));
	connect(&m_IntensitySlider, SIGNAL(valueChanged(int)), this, SLOT(OnIntensityChanged(int)));

	// Setup connections for opacity
	connect(&m_OpacitySlider, SIGNAL(valueChanged(int)), &m_OpacitySpinBox, SLOT(setValue(int)));
	connect(&m_OpacitySpinBox, SIGNAL(valueChanged(int)), &m_OpacitySlider, SLOT(setValue(int)));
	connect(&m_OpacitySlider, SIGNAL(valueChanged(int)), this, SLOT(OnOpacityChanged(int)));

	// Respond to changes in node selection
	connect(&gTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));

	SetupSelectionUI();
	SetupIntensityUI();
	SetupOpacityUI();
	SetupColorUI();

	OnNodeSelectionChanged(NULL);
}

void QNodePropertiesWidget::OnNodeSelectionChanged(QNode* pNode)
{
	if (!pNode)
	{
		setEnabled(false);
	}
	else
	{
		setEnabled(true);

		SetupSelectionUI();
		SetupIntensityUI();
		SetupOpacityUI();
		SetupColorUI();

		
		// Remove existing connections
		for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
		{
			QNode& Node = gTransferFunction.GetNode(i);

			disconnect(&Node, SIGNAL(IntensityChanged(QNode*)), this, SLOT(OnNodeIntensityChanged(QNode*)));
			disconnect(&Node, SIGNAL(OpacityChanged(QNode*)), this, SLOT(OnNodeOpacityChanged(QNode*)));
			disconnect(&Node, SIGNAL(ColorChanged(QNode*)), this, SLOT(OnNodeColorChanged(QNode*)));
		}

		// Setup connections
		connect(pNode, SIGNAL(IntensityChanged(QNode*)), this, SLOT(OnNodeIntensityChanged(QNode*)));
		connect(pNode, SIGNAL(OpacityChanged(QNode*)), this, SLOT(OnNodeOpacityChanged(QNode*)));
		connect(pNode, SIGNAL(ColorChanged(QNode*)), this, SLOT(OnNodeColorChanged(QNode*)));
	}
}

void QNodePropertiesWidget::OnNodeSelectionChanged(const int& Index)
{
	gTransferFunction.SetSelectedNode(Index);
	SetupSelectionUI();
}

void QNodePropertiesWidget::OnPreviousNode(void)
{
	gTransferFunction.SelectPreviousNode();
}

void QNodePropertiesWidget::OnNextNode(void)
{
	gTransferFunction.SelectNextNode();
}

void QNodePropertiesWidget::OnDeleteNode(void)
{
	if (!gTransferFunction.GetSelectedNode())
		return;

	gTransferFunction.RemoveNode(gTransferFunction.GetSelectedNode());
}

void QNodePropertiesWidget::OnIntensityChanged(const int& Position)
{
	if (gTransferFunction.GetSelectedNode())
		gTransferFunction.GetSelectedNode()->SetIntensity(Position);
}

void QNodePropertiesWidget::OnOpacityChanged(const int& Opacity)
{
	if (gTransferFunction.GetSelectedNode())
		gTransferFunction.GetSelectedNode()->SetOpacity(0.01f * Opacity);
}

void QNodePropertiesWidget::OnColorChanged(const QColor& Color)
{
	m_NodeSelection.clear();

	for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
		m_NodeSelection.addItem("Node " + QString::number(i + 1));
}

void QNodePropertiesWidget::OnNodeIntensityChanged(QNode* pNode)
{
	SetupIntensityUI();
}

void QNodePropertiesWidget::OnNodeOpacityChanged(QNode* pNode)
{
	if (pNode)
	{
		// Prevent circular dependency
		m_OpacitySlider.blockSignals(true);
		m_OpacitySpinBox.blockSignals(true);
		
		// Update values
		m_OpacitySlider.setValue(100.0f * pNode->GetOpacity());
		m_OpacitySpinBox.setValue(100.0f * pNode->GetOpacity());

		m_OpacitySlider.blockSignals(false);
		m_OpacitySpinBox.blockSignals(false);
	}
}

void QNodePropertiesWidget::OnNodeColorChanged(QNode* pNode)
{
//	if (pNode)
//		gTransferFunction.m_pSelectedNode->SetColor(pNode->GetColor());
}

void QNodePropertiesWidget::SetupSelectionUI(void)
{
	// Obtain selected node pointer
	QNode* pNode = gTransferFunction.GetSelectedNode();

	// Obtain current node index
	const int NodeIndex = gTransferFunction.GetNodes().indexOf(*gTransferFunction.GetSelectedNode());

	if (pNode && NodeIndex >= 0)
	{
		// Prevent circular dependency
		m_NodeSelection.blockSignals(true);

		m_NodeSelection.clear();

		for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
			m_NodeSelection.addItem("Node " + QString::number(i + 1));

		// Reflect node selection change in node selection combo box
		m_NodeSelection.setCurrentIndex(NodeIndex);

		m_NodeSelection.blockSignals(false);

		// Decide whether to enable/disable UI items
		const bool EnablePrevious	= NodeIndex > 0;
		const bool EnableNext		= NodeIndex < gTransferFunction.GetNodes().size() - 1;
		const bool EnableDelete		= gTransferFunction.GetSelectedNode() ? (NodeIndex != 0 && NodeIndex != gTransferFunction.GetNodes().size() - 1) : false;

		// Enable/disable buttons
		m_PreviousNode.setEnabled(EnablePrevious);
		m_NextNode.setEnabled(EnableNext);
		m_DeleteNode.setEnabled(EnableDelete);

		// Create tooltip strings
		QString PreviousToolTip = EnablePrevious ? "Select node " + QString::number(NodeIndex) : "No previous node";
		QString NextToolTip		= EnableNext ? "Select node " + QString::number(NodeIndex + 2) : "No next node";

		// Update push button tooltips
		m_PreviousNode.setStatusTip(PreviousToolTip);
		m_PreviousNode.setToolTip(PreviousToolTip);
		m_NextNode.setStatusTip(NextToolTip);
		m_NextNode.setToolTip(NextToolTip);
	}
}

void QNodePropertiesWidget::SetupIntensityUI(void)
{
	// Obtain selected node pointer
	QNode* pNode = gTransferFunction.GetSelectedNode();

	if (pNode)
	{
		// Obtain node index
		const int NodeIndex = gTransferFunction.GetNodes().indexOf(*gTransferFunction.GetSelectedNode());

		// Decide whether to enable the button
		const bool Enable = NodeIndex != 0 && NodeIndex != gTransferFunction.GetNodes().size() - 1;

		// Enable
		m_IntensityLabel.setEnabled(Enable);
		m_IntensitySlider.setEnabled(Enable);
		m_IntensitySpinBox.setEnabled(Enable);

		// Prevent circular dependency
		m_IntensitySlider.blockSignals(true);
		m_IntensitySpinBox.blockSignals(true);

		// Update values
		m_IntensitySlider.setRange(pNode->GetMinX(), pNode->GetMaxX());
		m_IntensitySlider.setValue(pNode->GetIntensity());
		m_IntensitySpinBox.setRange(pNode->GetMinX(), pNode->GetMaxX());
		m_IntensitySpinBox.setValue(pNode->GetIntensity());

		m_IntensitySlider.blockSignals(false);
		m_IntensitySpinBox.blockSignals(true);
	}
	else
	{
		// Enable
		m_IntensityLabel.setEnabled(false);
		m_IntensitySlider.setEnabled(false);
		m_IntensitySpinBox.setEnabled(false);
	}
}

void QNodePropertiesWidget::SetupOpacityUI(void)
{
	// Obtain selected node pointer
	QNode* pNode = gTransferFunction.GetSelectedNode();

	if (pNode)
	{
		// Enable
		m_OpacityLabel.setEnabled(true);
		m_OpacitySlider.setEnabled(true);
		m_OpacitySpinBox.setEnabled(true);

		// Prevent circular dependency
		m_OpacitySlider.blockSignals(true);
		m_OpacitySpinBox.blockSignals(true);

		// Update values
		m_OpacitySlider.setRange(100.0f * pNode->GetMinY(), 100.0f * pNode->GetMaxY());
		m_OpacitySlider.setValue(100.0f * pNode->GetOpacity());
		m_OpacitySpinBox.setRange(100.0f * pNode->GetMinY(), 100.0f * pNode->GetMaxY());
		m_OpacitySpinBox.setValue(100.0f * pNode->GetOpacity());

		m_OpacitySlider.blockSignals(false);
		m_OpacitySpinBox.blockSignals(true);
	}
	else
	{
		// Enable
		m_OpacityLabel.setEnabled(false);
		m_OpacitySlider.setEnabled(false);
		m_OpacitySpinBox.setEnabled(false);
	}
}

void QNodePropertiesWidget::SetupColorUI(void)
{

}


