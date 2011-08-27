
#include "NodePropertiesWidget.h"
#include "TransferFunction.h"
#include "NodeItem.h"

QNodePropertiesWidget::QNodePropertiesWidget(QWidget* pParent, QTransferFunction* pTransferFunction) :
	QWidget(pParent),
	m_pTransferFunction(pTransferFunction),
	m_pLastSelectedNode(NULL),
	m_pMainLayout(NULL),
	m_pSelectionLabel(NULL),
	m_pSelectionLayout(NULL),
	m_pNodeSelectionComboBox(NULL),
	m_pPreviousNodePushButton(NULL),
	m_pNextNodePushButton(NULL),
	m_pPositionLabel(NULL),
	m_pPositionSlider(NULL),
	m_pPositionSpinBox(NULL),
	m_pOpacityLabel(NULL),
	m_pOpacitySlider(NULL),
	m_pOpacitySpinBox(NULL),
	m_pColorLabel(NULL),
	m_pColorComboBox(NULL),
	m_pRoughnessLabel(NULL),
	m_pRoughnessSlider(NULL),
	m_pRoughnessSpinBox(NULL)
{
	setFixedHeight(100);
	
	// Node properties layout
	m_pMainLayout = new QGridLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
	m_pMainLayout->setContentsMargins(0, 0, 0, 0);

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

	m_pPreviousNodePushButton = new QPushButton("<<");
	m_pPreviousNodePushButton->setStatusTip("Select previous node");
	m_pPreviousNodePushButton->setToolTip("Select previous node");
	m_pPreviousNodePushButton->setFixedWidth(30);
	m_pPreviousNodePushButton->setFixedHeight(20);
	m_pPreviousNodePushButton->updateGeometry();
	m_pSelectionLayout->addWidget(m_pPreviousNodePushButton, 0, 1);

	connect(m_pPreviousNodePushButton, SIGNAL(pressed()), this, SLOT(OnPreviousNode()));

	m_pNextNodePushButton = new QPushButton(">>");
	m_pNextNodePushButton->setStatusTip("Select next node");
	m_pNextNodePushButton->setToolTip("Select next node");
	m_pNextNodePushButton->setFixedWidth(30);
	m_pNextNodePushButton->setFixedHeight(20);
	m_pSelectionLayout->addWidget(m_pNextNodePushButton, 0, 2);
	
	connect(m_pNextNodePushButton, SIGNAL(pressed()), this, SLOT(OnNextNode()));

	// Position
	m_pPositionLabel = new QLabel("Position");
	m_pPositionLabel->setStatusTip("Node position");
	m_pPositionLabel->setToolTip("Node position");
	m_pMainLayout->addWidget(m_pPositionLabel, 1, 0);

	m_pPositionSlider = new QSlider(Qt::Orientation::Horizontal);
	m_pPositionSlider->setStatusTip("Node position");
	m_pPositionSlider->setToolTip("Drag to change node position");
	m_pPositionSlider->setRange(0, 100);
	m_pMainLayout->addWidget(m_pPositionSlider, 1, 1);
	
	m_pPositionSpinBox = new QSpinBox;
	m_pPositionSpinBox->setStatusTip("Node position");
	m_pPositionSpinBox->setToolTip("Node position");
    m_pPositionSpinBox->setRange(0, 100);
	m_pMainLayout->addWidget(m_pPositionSpinBox, 1, 2);

	// Opacity
	m_pOpacityLabel = new QLabel("Opacity");
	m_pMainLayout->addWidget(m_pOpacityLabel, 2, 0);

	m_pOpacitySlider = new QSlider(Qt::Orientation::Horizontal);
	m_pOpacitySlider->setRange(0, 100);
	m_pMainLayout->addWidget(m_pOpacitySlider, 2, 1);
	
	m_pOpacitySpinBox = new QSpinBox;
    m_pOpacitySpinBox->setRange(0, 100);
	m_pMainLayout->addWidget(m_pOpacitySpinBox, 2, 2);
	
//	connect(m_pFocalDistanceSlider, SIGNAL(valueChanged(int)), m_pFocalDistanceSpinBox, SLOT(setValue(int)));
//	connect(m_pFocalDistanceSlider, SIGNAL(valueChanged(int)), this, SLOT(SetFocalDistance(int)));
//	connect(m_pFocalDistanceSpinBox, SIGNAL(valueChanged(int)), m_pFocalDistanceSlider, SLOT(setValue(int)));

	// Color
	m_pColorLabel = new QLabel("Color");
	m_pMainLayout->addWidget(m_pColorLabel, 3, 0);

	m_pColorComboBox = new QComboBox;
	m_pMainLayout->addWidget(m_pColorComboBox, 3, 1, 1, 2);

	/*
	// Roughness
	m_pRoughnessLabel = new QLabel("Roughness");
	m_pMainLayout->addWidget(m_pRoughnessLabel, 4, 0);

	m_pRoughnessSlider = new QSlider(Qt::Orientation::Horizontal);
	m_pRoughnessSlider->setRange(0, 100);
	m_pMainLayout->addWidget(m_pRoughnessSlider, 4, 1);
	
	m_pRoughnessSpinBox = new QSpinBox;
    m_pRoughnessSpinBox->setRange(0, 100);
	m_pMainLayout->addWidget(m_pRoughnessSpinBox, 4, 2);
	*/
	
	// Setup connections for position
	connect(m_pPositionSlider, SIGNAL(valueChanged(int)), m_pPositionSpinBox, SLOT(setValue(int)));
	connect(m_pPositionSpinBox, SIGNAL(valueChanged(int)), m_pPositionSlider, SLOT(setValue(int)));
	connect(m_pPositionSlider, SIGNAL(valueChanged(int)), this, SLOT(OnPositionChanged(int)));
//	connect(m_pTransferFunction, SIGNAL(FunctionChanged()), this, SLOT(OnTransferFunctionChanged()));

	// Setup connections for opacity
	connect(m_pOpacitySlider, SIGNAL(valueChanged(int)), m_pOpacitySpinBox, SLOT(setValue(int)));
	connect(m_pOpacitySpinBox, SIGNAL(valueChanged(int)), m_pOpacitySlider, SLOT(setValue(int)));
	connect(m_pOpacitySlider, SIGNAL(valueChanged(int)), this, SLOT(OnOpacityChanged(int)));

	// Respond to changes in node selection
	connect(m_pTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));
	
	// Respond to addition and removal of nodes
	connect(m_pTransferFunction, SIGNAL(NodeAdd(QNode*)), this, SLOT(OnNodeAdd(QNode*)));
	connect(m_pTransferFunction, SIGNAL(NodeRemove(QNode*)), this, SLOT(OnNodeRemove(QNode*)));
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

		m_pOpacitySlider->setValue(100.0f * pNode->GetOpacity());

		// Restrict the node's position
		m_pPositionSlider->setRange(pNode->GetMinX(), pNode->GetMaxX());
		m_pPositionSpinBox->setRange(pNode->GetMinX(), pNode->GetMaxX());

		// Obtain current node index
		const int CurrentNodeIndex = m_pTransferFunction->GetNodeIndex(pNode);

		// Reflect node selection change in node selection combo box
		m_pNodeSelectionComboBox->setCurrentIndex(CurrentNodeIndex);

		// Compute whether to enable/disable buttons
		const bool EnablePrevious	= CurrentNodeIndex > 0;
		const bool EnableNext		= CurrentNodeIndex < m_pTransferFunction->m_Nodes.size() - 1;
		const bool EnablePosition	= m_pTransferFunction->m_Nodes.front() != pNode && m_pTransferFunction->m_Nodes.back() != pNode;

		// Selectively enable/disable previous/next buttons
		m_pPreviousNodePushButton->setEnabled(EnablePrevious);
		m_pNextNodePushButton->setEnabled(EnableNext);

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

		// Disconnect previous node
		if (m_pLastSelectedNode)
		{
			disconnect(m_pLastSelectedNode, SIGNAL(PositionChanged(QNode*)), this, SLOT(OnNodePositionChanged(QNode*)));
			disconnect(m_pLastSelectedNode, SIGNAL(OpacityChanged(QNode*)), this, SLOT(OnNodeOpacityChanged(QNode*)));
			disconnect(m_pLastSelectedNode, SIGNAL(ColorChanged(QNode*)), this, SLOT(OnNodeColorChanged(QNode*)));
		}

		// Setup connections
		connect(pNode, SIGNAL(PositionChanged(QNode*)), this, SLOT(OnNodePositionChanged(QNode*)));
		connect(pNode, SIGNAL(OpacityChanged(QNode*)), this, SLOT(OnNodeOpacityChanged(QNode*)));
		connect(pNode, SIGNAL(ColorChanged(QNode*)), this, SLOT(OnNodeColorChanged(QNode*)));

		// Chache last node
		m_pLastSelectedNode = pNode;
	}
}

void QNodePropertiesWidget::OnNodeSelectionChanged(const int& Index)
{
	m_pTransferFunction->SetSelectedNode(Index);
}

void QNodePropertiesWidget::OnPreviousNode(void)
{
	m_pTransferFunction->SelectPreviousNode();
}

void QNodePropertiesWidget::OnNextNode(void)
{
	m_pTransferFunction->SelectNextNode();
}

void QNodePropertiesWidget::OnTransferFunctionChanged(void)
{
	if (m_pTransferFunction->m_pSelectedNode)
	{
		m_pPositionSlider->setValue(m_pTransferFunction->m_pSelectedNode->GetPosition());
//		m_pOpacitySlider->setValue(m_pTransferFunction->m_pSelectedNode->GetOpacity() * 100.0f);
	}
}

void QNodePropertiesWidget::OnPositionChanged(const int& Position)
{
	if (m_pTransferFunction->m_pSelectedNode)
	{
		m_pTransferFunction->m_pSelectedNode->SetPosition(Position);
//		m_pPositionSlider->setValue(Position);
	}
}

void QNodePropertiesWidget::OnOpacityChanged(const int& Opacity)
{
	if (m_pTransferFunction->m_pSelectedNode)
		m_pTransferFunction->m_pSelectedNode->SetOpacity(0.01f * Opacity);
}

void QNodePropertiesWidget::OnColorChanged(const QColor& Color)
{
	m_pNodeSelectionComboBox->clear();

	for (int i = 0; i < m_pTransferFunction->m_Nodes.size(); i++)
		m_pNodeSelectionComboBox->addItem("Node " + QString::number(i + 1));
}

void QNodePropertiesWidget::OnNodePositionChanged(QNode* pNode)
{
	if (pNode)
		m_pPositionSlider->setValue(pNode->GetPosition());
}

void QNodePropertiesWidget::OnNodeOpacityChanged(QNode* pNode)
{
	if (pNode)
		m_pOpacitySlider->setValue(pNode->GetOpacity());
}

void QNodePropertiesWidget::OnNodeColorChanged(QNode* pNode)
{
//	if (pNode)
//		m_pTransferFunction->m_pSelectedNode->SetColor(pNode->GetColor());
}

void QNodePropertiesWidget::OnNodeAdd(QNode* pNode)
{
	m_pNodeSelectionComboBox->clear();

	for (int i = 0; i < m_pTransferFunction->m_Nodes.size(); i++)
		m_pNodeSelectionComboBox->addItem("Node " + QString::number(i + 1));
}

void QNodePropertiesWidget::OnNodeRemove(QNode* pNode)
{
	/*
	if (m_pTransferFunction->m_pSelectedNode)
		m_pTransferFunction->m_pSelectedNode->SetOpacity(0.01f * Opacity);
	*/
}