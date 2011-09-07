
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
	m_DiffuseColor(),
	m_SpecularColor(),
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

	m_IntensitySlider.setOrientation(Qt::Horizontal);
	m_IntensitySlider.setStatusTip("Node position");
	m_IntensitySlider.setToolTip("Drag to change node position");
	m_MainLayout.addWidget(&m_IntensitySlider, 1, 1);
	
	m_IntensitySpinBox.setStatusTip("Node Position");
	m_IntensitySpinBox.setToolTip("Node Position");
	m_IntensitySpinBox.setSingleStep(1);
	m_MainLayout.addWidget(&m_IntensitySpinBox, 1, 2);

	connect(&m_IntensitySlider, SIGNAL(valueChanged(double)), &m_IntensitySpinBox, SLOT(setValue(double)));
	connect(&m_IntensitySpinBox, SIGNAL(valueChanged(double)), &m_IntensitySlider, SLOT(setValue(double)));
	connect(&m_IntensitySlider, SIGNAL(valueChanged(double)), this, SLOT(OnIntensityChanged(double)));

	// Opacity
	m_OpacityLabel.setText("Opacity");
	m_MainLayout.addWidget(&m_OpacityLabel, 2, 0);

	m_OpacitySlider.setOrientation(Qt::Horizontal);
	m_OpacitySlider.setStatusTip("Node Opacity");
	m_OpacitySlider.setToolTip("Node Opacity");
	m_OpacitySlider.setRange(0.0, 1.0);
	m_MainLayout.addWidget(&m_OpacitySlider, 2, 1);
	
	m_OpacitySpinBox.setStatusTip("Node Opacity");
	m_OpacitySpinBox.setToolTip("Node Opacity");
	m_OpacitySpinBox.setRange(0.0, 1.0);
	m_MainLayout.addWidget(&m_OpacitySpinBox, 2, 2);
	
	connect(&m_OpacitySlider, SIGNAL(valueChanged(double)), &m_OpacitySpinBox, SLOT(setValue(double)));
	connect(&m_OpacitySpinBox, SIGNAL(valueChanged(double)), &m_OpacitySlider, SLOT(setValue(double)));
	connect(&m_OpacitySlider, SIGNAL(valueChanged(double)), this, SLOT(OnOpacityChanged(double)));

	// Diffuse Color
	m_MainLayout.addWidget(new QLabel("Diffuse Color"), 3, 0);
	m_DiffuseColor.setFixedWidth(120);
	m_MainLayout.addWidget(&m_DiffuseColor, 3, 1);

	connect(&m_DiffuseColor, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnDiffuseColorChanged(const QColor&)));

	// Specular Color
	m_MainLayout.addWidget(new QLabel("Specular Color"), 4, 0);
	m_SpecularColor.setFixedWidth(120);
	m_MainLayout.addWidget(&m_SpecularColor, 4, 1);

	connect(&m_SpecularColor, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnSpecularColorChanged(const QColor&)));

	// Roughness
	m_RoughnessLabel.setText("Roughness");
	m_MainLayout.addWidget(&m_RoughnessLabel, 5, 0);

	m_RoughnessSlider.setOrientation(Qt::Horizontal);
	m_RoughnessSlider.setStatusTip("Roughness");
	m_RoughnessSlider.setToolTip("Roughness");
	m_RoughnessSlider.setRange(0.0f, 10000.0f);
	m_MainLayout.addWidget(&m_RoughnessSlider, 5, 1);

	m_RoughnessSpinBox.setStatusTip("Roughness");
	m_RoughnessSpinBox.setToolTip("Roughness");
	m_RoughnessSpinBox.setRange(0.0f, 10000.0f);
	m_MainLayout.addWidget(&m_RoughnessSpinBox, 5, 2);

	connect(&m_RoughnessSlider, SIGNAL(valueChanged(double)), &m_RoughnessSpinBox, SLOT(setValue(double)));
	connect(&m_RoughnessSpinBox, SIGNAL(valueChanged(double)), &m_RoughnessSlider, SLOT(setValue(double)));
	connect(&m_RoughnessSlider, SIGNAL(valueChanged(double)), this, SLOT(OnRoughnessChanged(double)));

	// Respond to changes in node selection
	connect(&gTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));

	SetupSelectionUI();

	OnNodeSelectionChanged(NULL);
}

void QNodePropertiesWidget::OnNodeSelectionChanged(QNode* pNode)
{
	if (!pNode)
	{
	}
	else
	{
		SetupSelectionUI();
		
		// Remove existing connections
		for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
		{
			QNode& Node = gTransferFunction.GetNode(i);

			disconnect(&Node, SIGNAL(IntensityChanged(QNode*)), this, SLOT(OnNodeIntensityChanged(QNode*)));
			disconnect(&Node, SIGNAL(OpacityChanged(QNode*)), this, SLOT(OnNodeOpacityChanged(QNode*)));
			disconnect(&Node, SIGNAL(DiffuseColorChanged(QNode*)), this, SLOT(OnNodeDiffuseColorChanged(QNode*)));
			disconnect(&Node, SIGNAL(SpecularColorChanged(QNode*)), this, SLOT(OnNodeSpecularColorChanged(QNode*)));
			disconnect(&Node, SIGNAL(RoughnessChanged(QNode*)), this, SLOT(OnNodeRoughnessChanged(QNode*)));
		}

		// Setup connections
		connect(pNode, SIGNAL(IntensityChanged(QNode*)), this, SLOT(OnNodeIntensityChanged(QNode*)));
		connect(pNode, SIGNAL(OpacityChanged(QNode*)), this, SLOT(OnNodeOpacityChanged(QNode*)));
		connect(pNode, SIGNAL(DiffuseColorChanged(QNode*)), this, SLOT(OnNodeDiffuseColorChanged(QNode*)));
		connect(pNode, SIGNAL(SpecularColorChanged(QNode*)), this, SLOT(OnNodeSpecularColorChanged(QNode*)));
		connect(pNode, SIGNAL(RoughnessChanged(QNode*)), this, SLOT(OnNodeRoughnessChanged(QNode*)));
	}

	OnNodeIntensityChanged(pNode);
	OnNodeOpacityChanged(pNode);
	OnNodeDiffuseColorChanged(pNode);
	OnNodeSpecularColorChanged(pNode);
	OnNodeRoughnessChanged(pNode);
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

void QNodePropertiesWidget::OnIntensityChanged(const double& Position)
{
	if (gTransferFunction.GetSelectedNode())
		gTransferFunction.GetSelectedNode()->SetIntensity(Position);
}

void QNodePropertiesWidget::OnOpacityChanged(const double& Opacity)
{
	if (gTransferFunction.GetSelectedNode())
		gTransferFunction.GetSelectedNode()->SetOpacity(Opacity);
}

void QNodePropertiesWidget::OnDiffuseColorChanged(const QColor& DiffuseColor)
{
	if (gTransferFunction.GetSelectedNode())
		gTransferFunction.GetSelectedNode()->SetDiffuseColor(DiffuseColor);
}

void QNodePropertiesWidget::OnSpecularColorChanged(const QColor& SpecularColor)
{
	if (gTransferFunction.GetSelectedNode())
		gTransferFunction.GetSelectedNode()->SetSpecularColor(SpecularColor);
}

void QNodePropertiesWidget::OnRoughnessChanged(const double& Roughness)
{
	if (gTransferFunction.GetSelectedNode())
		gTransferFunction.GetSelectedNode()->SetRoughness(Roughness);
}

void QNodePropertiesWidget::OnNodeIntensityChanged(QNode* pNode)
{
	bool Enable = false;

	if (pNode)
	{
		const int NodeIndex = gTransferFunction.GetNodes().indexOf(*pNode);

		Enable = NodeIndex != 0 && NodeIndex != gTransferFunction.GetNodes().size() - 1;

		m_IntensitySlider.setRange(pNode->GetMinX(), pNode->GetMaxX());
		m_IntensitySpinBox.setRange(pNode->GetMinX(), pNode->GetMaxX());

		m_IntensitySlider.setValue((double)pNode->GetIntensity(), true);
		m_IntensitySpinBox.setValue(pNode->GetIntensity(), true);
	}

	m_IntensityLabel.setEnabled(Enable);
	m_IntensitySlider.setEnabled(Enable);
	m_IntensitySpinBox.setEnabled(Enable);
}

void QNodePropertiesWidget::OnNodeOpacityChanged(QNode* pNode)
{
	const bool Enable = pNode != NULL;

	if (pNode)
	{
		m_OpacitySlider.setRange(pNode->GetMinY(), pNode->GetMaxY());
		m_OpacitySpinBox.setRange(pNode->GetMinY(), pNode->GetMaxY());

		m_OpacitySlider.setValue((double)pNode->GetOpacity(), true);
		m_OpacitySpinBox.setValue((double)pNode->GetOpacity(), true);
	}

	m_OpacityLabel.setEnabled(Enable);
	m_OpacitySlider.setEnabled(Enable);
	m_OpacitySpinBox.setEnabled(Enable);
}

void QNodePropertiesWidget::OnNodeDiffuseColorChanged(QNode* pNode)
{
	const bool Enable = pNode != NULL;

	if (pNode)
	{
		m_DiffuseColor.SetColor(pNode->GetDiffuseColor(), true);
	}

	m_DiffuseColor.setEnabled(Enable);
}

void QNodePropertiesWidget::OnNodeSpecularColorChanged(QNode* pNode)
{
	const bool Enable = pNode != NULL;

	if (pNode)
	{
		m_SpecularColor.SetColor(pNode->GetSpecularColor(), true);
	}

	m_SpecularColor.setEnabled(Enable);
}

void QNodePropertiesWidget::OnNodeRoughnessChanged(QNode* pNode)
{
	const bool Enable = pNode != NULL;

	if (pNode)
	{
		m_RoughnessSlider.setValue((double)pNode->GetRoughness(), true);
		m_RoughnessSpinBox.setValue((double)pNode->GetRoughness(), true);
	}

	m_RoughnessLabel.setEnabled(Enable);
	m_RoughnessSlider.setEnabled(Enable);
	m_RoughnessSpinBox.setEnabled(Enable);
}

void QNodePropertiesWidget::SetupSelectionUI(void)
{
	QNode* pNode = gTransferFunction.GetSelectedNode();

	if (pNode)
	{
		// Obtain current node index
		const int NodeIndex = gTransferFunction.GetNodes().indexOf(*gTransferFunction.GetSelectedNode());

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