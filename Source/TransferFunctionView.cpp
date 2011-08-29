
#include "TransferFunction.h"
#include "TransferFunctionView.h"
#include "TransferFunctionCanvas.h"
#include "TransferFunctionGradient.h"
#include "NodeItem.h"
#include "Scene.h"

QTransferFunctionView::QTransferFunctionView(QWidget* pParent) :
	QGraphicsView(pParent),
	m_pGraphicsScene(NULL),
	m_pTransferFunctionCanvas(NULL),
	m_pTransferFunctionGradient(NULL),
	m_MarginTop(8.0f),
	m_MarginBottom(36.0f),
	m_MarginLeft(15.0f),
	m_MarginRight(8.0f),
	m_AxisLabelX(NULL),
	m_AxisLabelY(NULL)
{
	// Dimensions
	setFixedHeight(300);

	// Styling
	setFrameShadow(Sunken);
	setFrameShape(NoFrame);

	// Never show scrollbars
	setVerticalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);
	setHorizontalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);

	// Status and tooltip
	setStatusTip("Transfer function editor");
	setToolTip("Transfer function editor");

	// Create scene and apply
	m_pGraphicsScene = new QGraphicsScene(this);
	setScene(m_pGraphicsScene);

	// Turn antialiasing on
	setRenderHint(QPainter::Antialiasing);

	// Respond to changes in the transfer function
	connect(&gTransferFunction, SIGNAL(FunctionChanged()), this, SLOT(Update()));

	// Create the transfer function canvas and add it to the scene
	m_pTransferFunctionCanvas = new QTransferFunctionCanvas(NULL, m_pGraphicsScene);
	m_pTransferFunctionCanvas->translate(m_MarginLeft, m_MarginTop);

	m_pTransferFunctionGradient = new QTransferFunctionGradient(NULL, m_pGraphicsScene);
	m_pTransferFunctionGradient->translate(m_MarginLeft, rect().bottom() - m_MarginBottom + 8);

//	m_pTransferFunctionCanvas->setVisible(false);

	// Respond to changes in node selection
	connect(&gTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));
	connect(&gTransferFunction, SIGNAL(HistogramChanged(void)), this, SLOT(OnHistogramChanged(void)));

	// X-axis label
	m_AxisLabelX = new QAxisLabel(NULL, "Density");
	m_pGraphicsScene->addItem(m_AxisLabelX);

	// Y-axis label
	m_AxisLabelY = new QAxisLabel(NULL, "Opacity");
	m_pGraphicsScene->addItem(m_AxisLabelY);
}

void QTransferFunctionView::drawBackground(QPainter* pPainter, const QRectF& Rectangle)
{
	QGraphicsView::drawBackground(pPainter, Rectangle);

	setBackgroundBrush(QBrush(QColor(240, 240, 240)));
}

void QTransferFunctionView::Update(void)
{
	m_pTransferFunctionCanvas->Update();
	m_pTransferFunctionGradient->Update();

	if (gpScene == NULL)
		return;

	gpScene->m_TransferFunctions.m_Kd.m_NoNodes = gTransferFunction.m_Nodes.size();
	gpScene->m_TransferFunctions.m_Ks.m_NoNodes = gTransferFunction.m_Nodes.size();
	gpScene->m_TransferFunctions.m_Kt.m_NoNodes = gTransferFunction.m_Nodes.size();

	for (int i = 0; i < gTransferFunction.m_Nodes.size(); i++)
	{
		QNode* pNode = gTransferFunction.m_Nodes[i];

		gpScene->m_TransferFunctions.m_Kd.m_P[i] = pNode->GetPosition();
		gpScene->m_TransferFunctions.m_Ks.m_P[i] = pNode->GetPosition();
		gpScene->m_TransferFunctions.m_Kt.m_P[i] = pNode->GetPosition();

		float Col = pNode->GetOpacity() * ((float)pNode->GetColor().red() / 255.0f);

		gpScene->m_TransferFunctions.m_Kd.m_C[i] = CColorRgbHdr(Col, Col, Col);
		gpScene->m_TransferFunctions.m_Ks.m_C[i] = CColorRgbHdr(Col, Col, Col);
		gpScene->m_TransferFunctions.m_Kt.m_C[i] = CColorRgbHdr(Col, Col, Col);
	}

	gpScene->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

void QTransferFunctionView::OnNodeSelectionChanged(QNode* pNode)
{
	// Deselect all nodes
	foreach (QNodeItem* pNode, m_pTransferFunctionCanvas->m_NodeItems)
		pNode->setSelected(false);

	if (pNode)
	{
		// Obtain node index
		const int NodeIndex = gTransferFunction.GetNodeIndex(pNode);

		// Select the node
		if (NodeIndex >= 0 && NodeIndex < m_pTransferFunctionCanvas->m_NodeItems.size())
		{
			m_pTransferFunctionCanvas->m_NodeItems[NodeIndex]->setSelected(true);
		}
	}
}

void QTransferFunctionView::OnHistogramChanged(void)
{
	m_pTransferFunctionCanvas->UpdateHistogram();
}

void QTransferFunctionView::resizeEvent(QResizeEvent* pResizeEvent)
{
	QGraphicsView::resizeEvent(pResizeEvent);

	setSceneRect(rect());

	QRectF CanvasRect = m_pTransferFunctionCanvas->rect();

	CanvasRect.setWidth(rect().width() - m_MarginLeft - m_MarginRight);
	CanvasRect.setHeight(rect().height() - m_MarginTop - m_MarginBottom);

	m_pTransferFunctionCanvas->setRect(CanvasRect);
	m_pTransferFunctionCanvas->Update();
	m_pTransferFunctionCanvas->UpdateHistogram();

	// Update transfer function gradient
	QRectF GradientRect = m_pTransferFunctionCanvas->rect();

	GradientRect.setWidth(rect().width() - m_MarginLeft - m_MarginRight);
	GradientRect.setHeight(12);

	m_pTransferFunctionGradient->setRect(GradientRect);
	m_pTransferFunctionGradient->Update();


	// Configure x-axis label
	m_AxisLabelX->setRect(QRectF(0, 0, CanvasRect.width(), 12));
	m_AxisLabelX->setX(m_MarginLeft);
	m_AxisLabelX->setY(m_MarginTop + CanvasRect.height() + 25);
	m_AxisLabelX->m_Text = "Intensity: [" + QString::number(gTransferFunction.m_RangeMin) + ", " + QString::number(gTransferFunction.m_RangeMax) + "]";

	// Configure y-axis label
	m_AxisLabelY->setRect(QRectF(0, 0, CanvasRect.height(), 8));
	m_AxisLabelY->setPos(0, m_MarginTop + CanvasRect.height());
	m_AxisLabelY->setRotation(-90.0f);
	m_AxisLabelY->m_Text = "Opacity (%): [0 - 100]";
}

void QTransferFunctionView::mousePressEvent(QMouseEvent* pEvent)
{
	QGraphicsView::mousePressEvent(pEvent);

	// Get node item under cursor
	QNodeItem* pNodeItem = dynamic_cast<QNodeItem*>(scene()->itemAt(pEvent->posF()));

	if (!pNodeItem)
	{
		// Add a new node if the user clicked the left button
		if (pEvent->button() == Qt::MouseButton::LeftButton && m_pTransferFunctionCanvas->rect().contains(pEvent->posF() - QPointF(m_MarginLeft, m_MarginTop)))
		{
			// Convert picked position to transfer function coordinates
			QPointF TfPoint = m_pTransferFunctionCanvas->SceneToTransferFunction(pEvent->posF() - QPointF(m_MarginLeft, m_MarginTop));

			// Generate random color
			int R = (int)(((float)rand() / (float)RAND_MAX) * 255.0f);
			int G = (int)(((float)rand() / (float)RAND_MAX) * 255.0f);
			int B = (int)(((float)rand() / (float)RAND_MAX) * 255.0f);

			// Create new transfer function node
			QNode* pNode = new QNode(&gTransferFunction, TfPoint.x(), TfPoint.y(), QColor(R, G, B, 255));

			// Add to node list
			gTransferFunction.AddNode(pNode);

			// Redraw
			m_pTransferFunctionCanvas->Update();

			// Select it immediately
			gTransferFunction.SetSelectedNode(pNode);
		}

		if (pEvent->button() == Qt::MouseButton::RightButton)
		{
			// Other wise no node selection
			gTransferFunction.SetSelectedNode((QNode*)NULL);
		}
	}
	else
	{
		if (pEvent->button() == Qt::MouseButton::LeftButton)
		{
			gTransferFunction.SetSelectedNode(pNodeItem->m_pNode);
		}
		else if (pEvent->button() == Qt::MouseButton::RightButton)
		{
			// Remove transfer function node if not the first or last node
			if (pNodeItem->m_pNode != gTransferFunction.m_Nodes.front() && pNodeItem->m_pNode != gTransferFunction.m_Nodes.back())
				gTransferFunction.RemoveNode(pNodeItem->m_pNode);
		}
	}
}