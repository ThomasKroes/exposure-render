
#include "NodePropertiesWidget.h"
#include "TransferFunction.h"
#include "NodeItem.h"
#include "ColorSelectorWidget.h"

QNodePropertiesWidget::QNodePropertiesWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_pLastSelectedNode(NULL),
	m_pMainLayout(NULL),
	m_pSelectionLabel(NULL),
	m_pSelectionLayout(NULL),
	m_pNodeSelectionComboBox(NULL),
	m_pPreviousNodePushButton(NULL),
	m_pNextNodePushButton(NULL),
	m_pDeleteNodePushButton(NULL),
	m_pPositionLabel(NULL),
	m_pPositionSlider(NULL),
	m_pPositionSpinBox(NULL),
	m_pOpacityLabel(NULL),
	m_pOpacitySlider(NULL),
	m_pOpacitySpinBox(NULL),
	m_pColorSelector(NULL),
	m_pRoughnessLabel(NULL),
	m_pRoughnessSlider(NULL),
	m_pRoughnessSpinBox(NULL)
{
	// Title, status and tooltip
	setTitle("Node Properties");
	setToolTip("Node Properties");
	setStatusTip("Node Properties");

	// Node properties layout
	m_pMainLayout = new QGridLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
//	m_pMainLayout->setContentsMargins(0, 0, 0, 0);

	setLayout(m_pMainLayout);

	// Node selection
	m_pSelectionLabel = new QLabel("Selection");
	m_pSelectionLabel->setStatusTip("Node selection");
	m_pSelectionLabel->setToolTip("Node selection");
	m_pMainLayout->addWidget(m_pSelectionLabel, 0, 0);

	m_pSelectionLayout = new QGridLayout();
	m_pSelectionLayout->setAlignment(Qt::AlignTop);
	m_pSelectionLayout->setContentsMargins(0, 0, 0, 0);
	
	m_pMainLayout->addLayout(m_pSelectionLayout, 0, 1, 1, 2);

	m_pNodeSelectionComboBox = new QComboBox;
	m_pNodeSelectionComboBox->setStatusTip("Node selection");
	m_pNodeSelectionComboBox->setToolTip("Node selection");
	m_pSelectionLayout->addWidget(m_pNodeSelectionComboBox, 0, 0);

	connect(m_pNodeSelectionComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnNodeSelectionChanged(int)));

	// Previous node
	m_pPreviousNodePushButton = new QPushButton("<");
	m_pPreviousNodePushButton->setStatusTip("Select previous node");
	m_pPreviousNodePushButton->setToolTip("Select previous node");
	m_pPreviousNodePushButton->setFixedWidth(20);
	m_pPreviousNodePushButton->setFixedHeight(20);
	m_pPreviousNodePushButton->updateGeometry();
	m_pSelectionLayout->addWidget(m_pPreviousNodePushButton, 0, 1);

	connect(m_pPreviousNodePushButton, SIGNAL(pressed()), this, SLOT(OnPreviousNode()));

	// Next node
	m_pNextNodePushButton = new QPushButton(">");
	m_pNextNodePushButton->setStatusTip("Select next node");
	m_pNextNodePushButton->setToolTip("Select next node");
	m_pNextNodePushButton->setFixedWidth(20);
	m_pNextNodePushButton->setFixedHeight(20);
	m_pSelectionLayout->addWidget(m_pNextNodePushButton, 0, 2);
	
	connect(m_pNextNodePushButton, SIGNAL(pressed()), this, SLOT(OnNextNode()));

	// Delete node
	m_pDeleteNodePushButton = new QPushButton("X");
	m_pDeleteNodePushButton->setStatusTip("Delete selected node");
	m_pDeleteNodePushButton->setToolTip("Delete selected node");
	m_pDeleteNodePushButton->setFixedWidth(20);
	m_pDeleteNodePushButton->setFixedHeight(20);
	m_pSelectionLayout->addWidget(m_pDeleteNodePushButton, 0, 3);
	
	connect(m_pDeleteNodePushButton, SIGNAL(pressed()), this, SLOT(OnDeleteNode()));

	// Position
	m_pPositionLabel = new QLabel("Position");
	m_pPositionLabel->setStatusTip("Node position");
	m_pPositionLabel->setToolTip("Node position");
	m_pMainLayout->addWidget(m_pPositionLabel, 1, 0);

	m_pPositionSlider = new QSlider(Qt::Orientation::Horizontal);
	m_pPositionSlider->setStatusTip("Node position");
	m_pPositionSlider->setToolTip("Drag to change node position");
	m_pPositionSlider->setRange(0, 100);
	m_pPositionSlider->setSingleStep(1);
	m_pMainLayout->addWidget(m_pPositionSlider, 1, 1);
	
	m_pPositionSpinBox = new QSpinBox;
	m_pPositionSpinBox->setStatusTip("Node Position");
	m_pPositionSpinBox->setToolTip("Node Position");
    m_pPositionSpinBox->setRange(0, 100);
	m_pPositionSpinBox->setSingleStep(1);
	m_pMainLayout->addWidget(m_pPositionSpinBox, 1, 2);

	// Opacity
	m_pOpacityLabel = new QLabel("Opacity");
	m_pMainLayout->addWidget(m_pOpacityLabel, 2, 0);

	m_pOpacitySlider = new QSlider(Qt::Orientation::Horizontal);
	m_pOpacitySlider->setStatusTip("Node Opacity");
	m_pOpacitySlider->setToolTip("Node Opacity");
	m_pOpacitySlider->setRange(0, 100);
	m_pOpacitySlider->setSingleStep(1);
	m_pMainLayout->addWidget(m_pOpacitySlider, 2, 1);
	
	m_pOpacitySpinBox = new QSpinBox;
	m_pOpacitySpinBox->setStatusTip("Node Opacity");
	m_pOpacitySpinBox->setToolTip("Node Opacity");
    m_pOpacitySpinBox->setRange(0, 100);
	m_pOpacitySpinBox->setSingleStep(1);
	m_pMainLayout->addWidget(m_pOpacitySpinBox, 2, 2);
	
	// Setup connections for position
	connect(m_pPositionSlider, SIGNAL(valueChanged(int)), m_pPositionSpinBox, SLOT(setValue(int)));
	connect(m_pPositionSpinBox, SIGNAL(valueChanged(int)), m_pPositionSlider, SLOT(setValue(int)));
	connect(m_pPositionSlider, SIGNAL(valueChanged(int)), this, SLOT(OnPositionChanged(int)));

	// Setup connections for opacity
	connect(m_pOpacitySlider, SIGNAL(valueChanged(int)), m_pOpacitySpinBox, SLOT(setValue(int)));
	connect(m_pOpacitySpinBox, SIGNAL(valueChanged(int)), m_pOpacitySlider, SLOT(setValue(int)));
	connect(m_pOpacitySlider, SIGNAL(valueChanged(int)), this, SLOT(OnOpacityChanged(int)));

	// Respond to changes in node selection
	connect(&gTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));
	
	// Respond to addition and removal of nodes
	connect(&gTransferFunction, SIGNAL(NodeAdd(QNode*)), this, SLOT(OnNodeAdd(QNode*)));
	connect(&gTransferFunction, SIGNAL(NodeRemove(QNode*)), this, SLOT(OnNodeRemove(QNode*)));
	connect(&gTransferFunction, SIGNAL(NodeRemoved(QNode*)), this, SLOT(OnNodeRemoved(QNode*)));
	
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

		// Suspend signals
		m_pPositionSlider->blockSignals(true);
		m_pPositionSpinBox->blockSignals(true);

		m_pPositionSlider->setRange(pNode->GetMinX(), pNode->GetMaxX());
		m_pPositionSlider->setValue(pNode->GetIntensity());
		m_pPositionSpinBox->setRange(pNode->GetMinX(), pNode->GetMaxX());
		m_pPositionSpinBox->setValue(pNode->GetIntensity());

		// Allow signals again
		m_pPositionSlider->blockSignals(false);
		m_pPositionSpinBox->blockSignals(false);

		// Suspend signals
		m_pOpacitySlider->blockSignals(true);
		m_pOpacitySpinBox->blockSignals(true);

		m_pOpacitySlider->setValue(100.0f * pNode->GetOpacity());

		// Allow signals again
		m_pOpacitySlider->blockSignals(false);
		m_pOpacitySpinBox->blockSignals(false);

		// Obtain current node index
		const int CurrentNodeIndex = gTransferFunction.GetNodeIndex(pNode);

		m_pNodeSelectionComboBox->blockSignals(true);

		// Reflect node selection change in node selection combo box
//		m_pNodeSelectionComboBox->setCurrentIndex(CurrentNodeIndex);

		m_pNodeSelectionComboBox->blockSignals(false);

		const int NodeIndex = gTransferFunction.GetNodes().indexOf(*gTransferFunction.GetSelectedNode());

		// Compute whether to enable/disable buttons
		const bool EnablePrevious	= CurrentNodeIndex > 0;
		const bool EnableNext		= CurrentNodeIndex < gTransferFunction.GetNodes().size() - 1;
		const bool EnablePosition	= NodeIndex != 0 && NodeIndex != gTransferFunction.GetNodes().size() - 1;
		const bool EnableDelete		= gTransferFunction.GetSelectedNode() ? (NodeIndex != 0 && NodeIndex != gTransferFunction.GetNodes().size() - 1) : false;
		
		// Enable/disable buttons
		m_pPreviousNodePushButton->setEnabled(EnablePrevious);
		m_pNextNodePushButton->setEnabled(EnableNext);
		m_pDeleteNodePushButton->setEnabled(EnableDelete);

		// Enable/disable position label, slider and spinbox
		m_pPositionLabel->setEnabled(EnablePosition);
		m_pPositionSlider->setEnabled(EnablePosition);
		m_pPositionSpinBox->setEnabled(EnablePosition);

		// Create tooltip strings
		QString PreviousToolTip = EnablePrevious ? "Select node " + QString::number(CurrentNodeIndex) : "No previous node";
		QString NextToolTip		= EnableNext ? "Select node " + QString::number(CurrentNodeIndex + 2) : "No next node";

		// Update push button tooltips
		m_pPreviousNodePushButton->setStatusTip(PreviousToolTip);
		m_pPreviousNodePushButton->setToolTip(PreviousToolTip);
		m_pNextNodePushButton->setStatusTip(NextToolTip);
		m_pNextNodePushButton->setToolTip(NextToolTip);

		// Setup connections
		connect(pNode, SIGNAL(PositionChanged(QNode*)), this, SLOT(OnNodePositionChanged(QNode*)));
		connect(pNode, SIGNAL(OpacityChanged(QNode*)), this, SLOT(OnNodeOpacityChanged(QNode*)));
		connect(pNode, SIGNAL(ColorChanged(QNode*)), this, SLOT(OnNodeColorChanged(QNode*)));

		// Cache last node
		m_pLastSelectedNode = pNode;
	}
}

void QNodePropertiesWidget::OnNodeSelectionChanged(const int& Index)
{
//	gTransferFunction.SetSelectedNode(Index);
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

void QNodePropertiesWidget::OnPositionChanged(const int& Position)
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
	m_pNodeSelectionComboBox->clear();

	for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
		m_pNodeSelectionComboBox->addItem("Node " + QString::number(i + 1));
}

void QNodePropertiesWidget::OnNodePositionChanged(QNode* pNode)
{
	if (pNode)
	{
		// Prevent circular dependency
		m_pPositionSlider->blockSignals(true);
		m_pPositionSpinBox->blockSignals(true);

		// Update values
		m_pPositionSlider->setValue(pNode->GetIntensity());
		m_pPositionSpinBox->setValue(pNode->GetIntensity());

		m_pPositionSlider->blockSignals(false);
		m_pPositionSpinBox->blockSignals(true);
	}
}

void QNodePropertiesWidget::OnNodeOpacityChanged(QNode* pNode)
{
	if (pNode)
	{
		// Prevent circular dependency
		m_pOpacitySlider->blockSignals(true);
		m_pOpacitySpinBox->blockSignals(true);
		
		// Update values
		m_pOpacitySlider->setValue(100.0f * pNode->GetOpacity());
		m_pOpacitySpinBox->setValue(100.0f * pNode->GetOpacity());

		m_pOpacitySlider->blockSignals(false);
		m_pOpacitySpinBox->blockSignals(false);
	}
}

void QNodePropertiesWidget::OnNodeColorChanged(QNode* pNode)
{
//	if (pNode)
//		gTransferFunction.m_pSelectedNode->SetColor(pNode->GetColor());
}

void QNodePropertiesWidget::OnNodeAdd(QNode* pNode)
{
	m_pNodeSelectionComboBox->clear();

	for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
		m_pNodeSelectionComboBox->addItem("Node " + QString::number(i + 1));
}

void QNodePropertiesWidget::OnNodeRemove(QNode* pNode)
{
	// Disconnect previous node
	if (m_pLastSelectedNode)
	{
		disconnect(m_pLastSelectedNode, SIGNAL(PositionChanged(QNode*)), this, SLOT(OnNodePositionChanged(QNode*)));
		disconnect(m_pLastSelectedNode, SIGNAL(OpacityChanged(QNode*)), this, SLOT(OnNodeOpacityChanged(QNode*)));
		disconnect(m_pLastSelectedNode, SIGNAL(ColorChanged(QNode*)), this, SLOT(OnNodeColorChanged(QNode*)));

		m_pLastSelectedNode = NULL;
	}
}

void QNodePropertiesWidget::OnNodeRemoved(QNode* pNode)
{
	m_pNodeSelectionComboBox->clear();

	for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
		m_pNodeSelectionComboBox->addItem("Node " + QString::number(i + 1));
}