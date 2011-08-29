
#include "TransferFunction.h"
#include "TransferFunctionView.h"
#include "TransferFunctionCanvas.h"
#include "NodeItem.h"
#include "Scene.h"

QTransferFunctionView::QTransferFunctionView(QWidget* pParent) :
	QGraphicsView(pParent),
	m_pGraphicsScene(NULL),
	m_pTransferFunctionCanvas(NULL),
	m_Margin(24.0f),
	m_AxisLabelX(NULL),
	m_AxisLabelY(NULL),
	m_pMinX(NULL),
	m_pMaxX(NULL)
{
	// Dimensions
	setFixedHeight(250);

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
	m_pTransferFunctionCanvas->translate(m_Margin, m_Margin);
//	m_pTransferFunctionCanvas->setVisible(false);

	// Respond to changes in node selection
	connect(&gTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));

	// X-axis label
	m_AxisLabelX = new QAxisLabel(NULL, "Density");
	m_pGraphicsScene->addItem(m_AxisLabelX);

	// Y-axis label
	m_AxisLabelY = new QAxisLabel(NULL, "Opacity");
	m_pGraphicsScene->addItem(m_AxisLabelY);

	// Min x label
	m_pMinX = new QAxisLabel(NULL, QString::number(gTransferFunction.m_RangeMin));
	m_pGraphicsScene->addItem(m_pMinX);
	
	// Max x label
	m_pMaxX = new QAxisLabel(NULL, QString::number(gTransferFunction.m_RangeMax));
	m_pGraphicsScene->addItem(m_pMaxX);
}

void QTransferFunctionView::drawBackground(QPainter* pPainter, const QRectF& Rectangle)
{
	QGraphicsView::drawBackground(pPainter, Rectangle);

	setBackgroundBrush(QBrush(QColor(240, 240, 240)));
}

void QTransferFunctionView::Update(void)
{
	m_pTransferFunctionCanvas->Update();

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

	CanvasRect.setWidth(rect().width() - 2.0f * m_Margin);
	CanvasRect.setHeight(rect().height() - 2.0f * m_Margin);

	m_pTransferFunctionCanvas->setRect(CanvasRect);
	m_pTransferFunctionCanvas->Update();
	m_pTransferFunctionCanvas->UpdateHistogram();

	// Configure x-axis label
	m_AxisLabelX->setRect(QRectF(0, 0, CanvasRect.width(), 12));
	m_AxisLabelX->setX(m_Margin);
	m_AxisLabelX->setY(m_Margin + 12 + CanvasRect.height());

	// Configure y-axis label
	m_AxisLabelY->setRect(QRectF(0, 0, CanvasRect.height(), 8));
	m_AxisLabelY->setPos(0, m_Margin + CanvasRect.height());
	m_AxisLabelY->setRotation(-90.0f);

	// Min X
	m_pMinX->setRect(QRectF(0, 0, 20, 12));
	m_pMinX->setX(m_Margin - 10);
	m_pMinX->setY(m_Margin + CanvasRect.height() + 2);

	// Max X
	m_pMaxX->setRect(QRectF(0, 0, 20, 12));
	m_pMaxX->setX(m_Margin - 10 + CanvasRect.width());
	m_pMaxX->setY(m_Margin + CanvasRect.height() + 2);

	// Min Y
//	m_pMinY->setRect(QRectF(0, 0, 20, 12));
//	m_pMinY->setX(m_Margin - 10);
//	m_pMinY->setY(m_Margin + CanvasRect.height() + 2);

	// Max X
//	m_pMaxY->setRect(QRectF(0, 0, 20, 12));
//	m_pMaxY->setX(m_Margin - 10 + CanvasRect.width());
//	m_pMaxY->setY(m_Margin + CanvasRect.height() + 2);
}

void QTransferFunctionView::mousePressEvent(QMouseEvent* pEvent)
{
	QGraphicsView::mousePressEvent(pEvent);

	// Get node item under cursor
	QNodeItem* pNodeItem = dynamic_cast<QNodeItem*>(scene()->itemAt(pEvent->posF()));

	if (!pNodeItem)
	{
		// Add a new node if the user clicked the left button
		if (pEvent->button() == Qt::MouseButton::LeftButton && m_pTransferFunctionCanvas->rect().contains(pEvent->posF() - QPointF(m_Margin, m_Margin)))
		{
			// Convert picked position to transfer function coordinates
			QPointF TfPoint = m_pTransferFunctionCanvas->SceneToTransferFunction(pEvent->posF() - QPointF(m_Margin, m_Margin));

			// Generate random color
			int R = (int)(((float)rand() / (float)RAND_MAX) * 255.0f);
			int G = (int)(((float)rand() / (float)RAND_MAX) * 255.0f);
			int B = (int)(((float)rand() / (float)RAND_MAX) * 255.0f);

			// Create new transfer function node
			QNode* pNode = new QNode(TfPoint.x(), TfPoint.y(), QColor(R, G, B, 255));

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