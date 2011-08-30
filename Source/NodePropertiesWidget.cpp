
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
	m_pIntensityLabel(NULL),
	m_pIntensitySlider(NULL),
	m_pIntensitySpinBox(NULL),
	m_pOpacityLabel(NULL),
	m_pOpacitySlider(NULL),
	m_pOpacitySpinBox(NULL),
	m_pColorSelector(NULL),
	m_pRoughnessLabel(NULL),
	m_pRoughnessSlider(NULL),
	m_pRoughnessSpinBox(NULL)
{
	// Dimensions
//	setFixedHeight(150);
//	setFixedWidth(250);

	// Set size policy to minimum
//	setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);

	// Title, status and tooltip
	setTitle("Node Properties");
	setToolTip("Node Properties");
	setStatusTip("Node Properties");

	// Node properties layout
	m_pMainLayout = new QGridLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
//	m_pMainLayout->setContentsMargins(0, 0, 0, 0);

	setLayout(m_pMainLayout);

	setAlignment(Qt::AlignTop);

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
	m_pIntensityLabel = new QLabel("Position");
	m_pIntensityLabel->setStatusTip("Node position");
	m_pIntensityLabel->setToolTip("Node position");
	m_pMainLayout->addWidget(m_pIntensityLabel, 1, 0);

	m_pIntensitySlider = new QSlider(Qt::Orientation::Horizontal);
	m_pIntensitySlider->setStatusTip("Node position");
	m_pIntensitySlider->setToolTip("Drag to change node position");
	m_pIntensitySlider->setRange(0, 100);
	m_pIntensitySlider->setSingleStep(1);
	m_pMainLayout->addWidget(m_pIntensitySlider, 1, 1);
	
	m_pIntensitySpinBox = new QSpinBox;
	m_pIntensitySpinBox->setStatusTip("Node Position");
	m_pIntensitySpinBox->setToolTip("Node Position");
    m_pIntensitySpinBox->setRange(0, 100);
	m_pIntensitySpinBox->setSingleStep(1);
	m_pMainLayout->addWidget(m_pIntensitySpinBox, 1, 2);

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
	connect(m_pIntensitySlider, SIGNAL(valueChanged(int)), m_pIntensitySpinBox, SLOT(setValue(int)));
	connect(m_pIntensitySpinBox, SIGNAL(valueChanged(int)), m_pIntensitySlider, SLOT(setValue(int)));
	connect(m_pIntensitySlider, SIGNAL(valueChanged(int)), this, SLOT(OnIntensityChanged(int)));

	// Setup connections for opacity
	connect(m_pOpacitySlider, SIGNAL(valueChanged(int)), m_pOpacitySpinBox, SLOT(setValue(int)));
	connect(m_pOpacitySpinBox, SIGNAL(valueChanged(int)), m_pOpacitySlider, SLOT(setValue(int)));
	connect(m_pOpacitySlider, SIGNAL(valueChanged(int)), this, SLOT(OnOpacityChanged(int)));

	// Respond to changes in node selection
	connect(&gTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));
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

			disconnect(&Node, SIGNAL(PositionChanged(QNode*)), this, SLOT(OnNodeIntensityChanged(QNode*)));
			disconnect(&Node, SIGNAL(OpacityChanged(QNode*)), this, SLOT(OnNodeOpacityChanged(QNode*)));
			disconnect(&Node, SIGNAL(ColorChanged(QNode*)), this, SLOT(OnNodeColorChanged(QNode*)));
		}

		// Setup connections
		connect(pNode, SIGNAL(PositionChanged(QNode*)), this, SLOT(OnNodeIntensityChanged(QNode*)));
		connect(pNode, SIGNAL(OpacityChanged(QNode*)), this, SLOT(OnNodeOpacityChanged(QNode*)));
		connect(pNode, SIGNAL(ColorChanged(QNode*)), this, SLOT(OnNodeColorChanged(QNode*)));

		// Cache last node
		m_pLastSelectedNode = pNode;
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
	m_pNodeSelectionComboBox->clear();

	for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
		m_pNodeSelectionComboBox->addItem("Node " + QString::number(i + 1));
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

void QNodePropertiesWidget::SetupSelectionUI(void)
{
	// Obtain selected node pointer
	QNode* pNode = gTransferFunction.GetSelectedNode();

	// Obtain current node index
	const int NodeIndex = gTransferFunction.GetNodes().indexOf(*gTransferFunction.GetSelectedNode());

	if (pNode && NodeIndex >= 0)
	{
		// Prevent circular dependency
		m_pNodeSelectionComboBox->blockSignals(true);

		m_pNodeSelectionComboBox->clear();

		for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
			m_pNodeSelectionComboBox->addItem("Node " + QString::number(i + 1));

		// Reflect node selection change in node selection combo box
		m_pNodeSelectionComboBox->setCurrentIndex(NodeIndex);

		m_pNodeSelectionComboBox->blockSignals(false);

		// Decide whether to enable/disable UI items
		const bool EnablePrevious	= NodeIndex > 0;
		const bool EnableNext		= NodeIndex < gTransferFunction.GetNodes().size() - 1;
		const bool EnableDelete		= gTransferFunction.GetSelectedNode() ? (NodeIndex != 0 && NodeIndex != gTransferFunction.GetNodes().size() - 1) : false;

		// Enable/disable buttons
		m_pPreviousNodePushButton->setEnabled(EnablePrevious);
		m_pNextNodePushButton->setEnabled(EnableNext);
		m_pDeleteNodePushButton->setEnabled(EnableDelete);

		// Create tooltip strings
		QString PreviousToolTip = EnablePrevious ? "Select node " + QString::number(NodeIndex) : "No previous node";
		QString NextToolTip		= EnableNext ? "Select node " + QString::number(NodeIndex + 2) : "No next node";

		// Update push button tooltips
		m_pPreviousNodePushButton->setStatusTip(PreviousToolTip);
		m_pPreviousNodePushButton->setToolTip(PreviousToolTip);
		m_pNextNodePushButton->setStatusTip(NextToolTip);
		m_pNextNodePushButton->setToolTip(NextToolTip);
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
		m_pIntensityLabel->setEnabled(Enable);
		m_pIntensitySlider->setEnabled(Enable);
		m_pIntensitySpinBox->setEnabled(Enable);

		// Prevent circular dependency
		m_pIntensitySlider->blockSignals(true);
		m_pIntensitySpinBox->blockSignals(true);

		// Update values
		m_pIntensitySlider->setRange(pNode->GetMinX(), pNode->GetMaxX());
		m_pIntensitySlider->setValue(pNode->GetIntensity());
		m_pIntensitySpinBox->setRange(pNode->GetMinX(), pNode->GetMaxX());
		m_pIntensitySpinBox->setValue(pNode->GetIntensity());

		m_pIntensitySlider->blockSignals(false);
		m_pIntensitySpinBox->blockSignals(true);
	}
	else
	{
		// Enable
		m_pIntensityLabel->setEnabled(false);
		m_pIntensitySlider->setEnabled(false);
		m_pIntensitySpinBox->setEnabled(false);
	}
}

void QNodePropertiesWidget::SetupOpacityUI(void)
{
	// Obtain selected node pointer
	QNode* pNode = gTransferFunction.GetSelectedNode();

	if (pNode)
	{
		// Enable
		m_pOpacityLabel->setEnabled(true);
		m_pOpacitySlider->setEnabled(true);
		m_pOpacitySpinBox->setEnabled(true);

		// Prevent circular dependency
		m_pOpacitySlider->blockSignals(true);
		m_pOpacitySpinBox->blockSignals(true);

		// Update values
		m_pOpacitySlider->setRange(100.0f * pNode->GetMinY(), 100.0f * pNode->GetMaxY());
		m_pOpacitySlider->setValue(100.0f * pNode->GetOpacity());
		m_pOpacitySpinBox->setRange(100.0f * pNode->GetMinY(), 100.0f * pNode->GetMaxY());
		m_pOpacitySpinBox->setValue(100.0f * pNode->GetOpacity());

		m_pOpacitySlider->blockSignals(false);
		m_pOpacitySpinBox->blockSignals(true);
	}
	else
	{
		// Enable
		m_pOpacityLabel->setEnabled(false);
		m_pOpacitySlider->setEnabled(false);
		m_pOpacitySpinBox->setEnabled(false);
	}
}

void QNodePropertiesWidget::SetupColorUI(void)
{

}


