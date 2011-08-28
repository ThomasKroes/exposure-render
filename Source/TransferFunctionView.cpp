
#include "TransferFunctionView.h"
#include "TransferFunction.h"
#include "NodeItem.h"

QTransferFunctionCanvas::QTransferFunctionCanvas(QGraphicsItem* pParent, QGraphicsScene* pGraphicsScene) :
	QGraphicsRectItem(pParent, pGraphicsScene),
	m_BackgroundBrush(),
	m_BackgroundPen(),
	m_GridLinesHorizontal(),
	m_GridPenHorizontal(),
	m_GridPenVertical()
{
	// Create polygon graphics item
	m_pPolygon = new QGraphicsPolygonItem;
	m_pPolygon->setParentItem(this);

	// Background styling
	m_BackgroundBrush.setColor(QColor(210, 210, 210));
	m_BackgroundBrush.setStyle(Qt::BrushStyle::SolidPattern);
	m_BackgroundPen.setColor(QColor(90, 90, 90));
	m_BackgroundPen.setJoinStyle(Qt::MiterJoin);
	m_BackgroundPen.setWidthF(0.5f);
	m_BackgroundPen.setStyle(Qt::SolidLine);

	// Background styling
	setBrush(m_BackgroundBrush);
	setPen(m_BackgroundPen);

	// Make sure this rectangle is drawn behind everything else
	setZValue(0);

	// Grid
	m_GridPenHorizontal.setColor(QColor(100, 100, 100, 200));
	m_GridPenHorizontal.setWidthF(0.5f);
	m_GridPenVertical.setColor(QColor(100, 100, 100, 200));
	m_GridPenVertical.setWidthF(0.5f);

	// Update the canvas
	Update();
}

void QTransferFunctionCanvas::SetSelectedNode(QNode* pSelectedNode)
{
	gTransferFunction.SetSelectedNode(pSelectedNode);
}

void QTransferFunctionCanvas::Update(void)
{
	UpdateGrid();
	UpdateHistogram();
	UpdateNodes();
	UpdateEdges();
	UpdateGradient();
	UpdatePolygon();
}

void QTransferFunctionCanvas::UpdateGrid(void)
{
	// Horizontal grid lines
	const float DeltaY = 0.1f * rect().height();

	// Remove old horizontal grid lines
	foreach(QGraphicsLineItem* pLine, m_GridLinesHorizontal)
		scene()->removeItem(pLine);

	// Clear the edges list
	m_GridLinesHorizontal.clear();

	for (int i = 1; i < 10; i++)
	{
		// Create a new grid line
		QGraphicsLineItem* pLine = new QGraphicsLineItem(QLineF(0, i * DeltaY, rect().width(), i * DeltaY));

		pLine->setPen(m_GridPenHorizontal);
		pLine->setZValue(100);
		pLine->setParentItem(this);

		// Add it to the list so we can remove them from the canvas when needed
		m_GridLinesHorizontal.append(pLine);
	}
}

void QTransferFunctionCanvas::UpdateHistogram(void)
{
}

void QTransferFunctionCanvas::UpdateNodes(void)
{
	// Remove old nodes
	foreach(QNodeItem* pNodeItem, m_NodeItems)
		scene()->removeItem(pNodeItem);

	// Clear the node items list
	m_NodeItems.clear();

	// Create the node items
	foreach(QNode* pNode, gTransferFunction.m_Nodes)
	{
		// Create new node item
		QNodeItem* pNodeItem = new QNodeItem(NULL, pNode, this);

		// Make it child of us
		pNodeItem->setParentItem(this);

		// Compute node center in canvas coordinates
		QPointF NodeCenter = TransferFunctionToScene(QPointF(pNode->GetX(), pNode->GetY()));

		// Set it's position
		pNodeItem->setPos(NodeCenter);

		// Make sure node items are rendered on top
		pNodeItem->setZValue(50000);

		// Add it to the list so we can remove them from the canvas when needed
		m_NodeItems.append(pNodeItem);
	}
}

void QTransferFunctionCanvas::UpdateEdges(void)
{
	// Remove old edges
	foreach(QGraphicsLineItem* pLine, m_EdgeItems)
		scene()->removeItem(pLine);

	// Clear the edges list
	m_EdgeItems.clear();

	for (int i = 1; i < m_NodeItems.size(); i++)
	{
		QPointF PtFrom(m_NodeItems[i - 1]->pos());
		QPointF PtTo(m_NodeItems[i]->pos());

		QGraphicsLineItem* pLine = new QGraphicsLineItem(QLineF(PtFrom, PtTo));
		
		pLine->setParentItem(this);

		// Set the pen
		pLine->setPen(QPen(QColor(110, 110, 100), 0.5));

		// Ensure the item is drawn in the right order
		pLine->setZValue(300);

		m_EdgeItems.append(pLine);
	}
}

void QTransferFunctionCanvas::UpdateGradient(void)
{
	m_LinearGradient.setStart(0, 0);
	m_LinearGradient.setFinalStop(rect().right(), rect().top());

	QGradientStops GradientStops;

	// Set the gradient stops
	foreach(QNode* pNode, gTransferFunction.m_Nodes)
	{
		QColor Color = pNode->GetColor();

		// Clamp node opacity to obtain valid alpha for display
		float Alpha = qMin(1.0f, qMax(0.0f, pNode->GetOpacity()));

		Color.setAlphaF(0.5f);

		// Add a new gradient stop
		GradientStops.append(QGradientStop(pNode->GetNormalizedX(), Color));
	}

	m_LinearGradient.setStops(GradientStops);
}

void QTransferFunctionCanvas::UpdatePolygon(void)
{
	QPolygonF Polygon;

	// Set the gradient stops
	for (int i = 0; i < m_NodeItems.size(); i++)
	{
		QNodeItem* pNodeItem = m_NodeItems[i];

		// Compute polygon point in scene coordinates
		QPointF ScenePoint = pNodeItem->pos();

//		ScenePoint.setY(rect().height());

		if (pNodeItem == m_NodeItems.front())
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(rect().height());

			Polygon.append(CenterCopy);
		}

		Polygon.append(ScenePoint);

		if (pNodeItem == m_NodeItems.back())
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(rect().height());

			Polygon.append(CenterCopy);
		}
	}

	m_pPolygon->setZValue(5000);
	m_pPolygon->setPolygon(Polygon);
	m_pPolygon->setBrush(QBrush(m_LinearGradient));
	m_pPolygon->setPen(QPen(Qt::PenStyle::NoPen));
}

// Maps from scene coordinates to transfer function coordinates
QPointF QTransferFunctionCanvas::SceneToTransferFunction(const QPointF& ScenePoint)
{
	const float NormalizedX = ScenePoint.x() / (float)rect().width();
	const float NormalizedY = 1.0f - (ScenePoint.y() / (float)rect().height());

	const float TfX = gTransferFunction.m_RangeMin + NormalizedX * gTransferFunction.m_Range;
	const float TfY = NormalizedY;

	return QPointF(TfX, TfY);
}

// Maps from transfer function coordinates to scene coordinates
QPointF QTransferFunctionCanvas::TransferFunctionToScene(const QPointF& TfPoint)
{
	const float NormalizedX = (TfPoint.x() - gTransferFunction.m_RangeMin) / gTransferFunction.m_Range;
	const float NormalizedY = 1.0f - TfPoint.y();

	const float SceneX = NormalizedX * rect().width();
	const float SceneY = NormalizedY * rect().height();

	return QPointF(SceneX, SceneY);
}

QTransferFunctionView::QTransferFunctionView(QWidget* pParent) :
	QGraphicsView(pParent),
	m_pGraphicsScene(NULL),
	m_pTransferFunctionCanvas(NULL),
	m_Margin(12.0f),
	m_AxisLabelX(NULL),
	m_AxisLabelY(NULL)
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
	m_pTransferFunctionCanvas->setVisible(false);

	// Respond to changes in node selection
	connect(&gTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));

	// X-axis label
	m_AxisLabelX = new QAxisLabel(NULL, "Density");
	m_pGraphicsScene->addItem(m_AxisLabelX);

	// Y-axis label
	m_AxisLabelY = new QAxisLabel(NULL, "Density");
	m_pGraphicsScene->addItem(m_AxisLabelX);
}

void QTransferFunctionView::drawBackground(QPainter* pPainter, const QRectF& Rectangle)
{
	QGraphicsView::drawBackground(pPainter, Rectangle);

	setBackgroundBrush(QBrush(QColor(240, 240, 240)));


	pPainter->drawText(rect(), Qt::AlignCenter, "Qt");
}

void QTransferFunctionView::Update(void)
{
	m_pTransferFunctionCanvas->Update();
}

void QTransferFunctionView::OnNodeSelectionChanged(QNode* pNode)
{
	if (pNode)
	{
		// Deselect all nodes
		foreach (QNodeItem* pNode, m_pTransferFunctionCanvas->m_NodeItems)
			pNode->setSelected(false);

		// Obtain node index
		const int NodeIndex = gTransferFunction.GetNodeIndex(pNode);

		// Select the node
		if (NodeIndex >= 0)
		{
	//		m_pTransferFunctionCanvas->m_NodeItems[NodeIndex]->setSelected(true);
	//		m_pTransferFunctionCanvas->m_NodeItems[NodeIndex]->update();
		}
	}
	else
	{
	}
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
}

void QTransferFunctionView::mousePressEvent(QMouseEvent* pEvent)
{
	QGraphicsView::mousePressEvent(pEvent);

	// Get node item under cursor
	QNodeItem* pNodeItem = dynamic_cast<QNodeItem*>(scene()->itemAt(pEvent->posF()));

	if (!pNodeItem)
	{
		// Add a new node if the user clicked the left button
		if (pEvent->button() == Qt::MouseButton::LeftButton)
		{
			QPointF TfPoint = m_pTransferFunctionCanvas->SceneToTransferFunction(pEvent->pos());

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
//			m_pTransferFunctionCanvas->SetSelectedNode(pNode);
		}
		else
		{
			// Other wise no node selection
			m_pTransferFunctionCanvas->SetSelectedNode(NULL);
		}
	}
	else
	{
		if (pEvent->button() == Qt::MouseButton::LeftButton)
		{
			m_pTransferFunctionCanvas->SetSelectedNode(pNodeItem->m_pNode);
		}
		else if (pEvent->button() == Qt::MouseButton::RightButton)
		{
			// Remove transfer function node if not the first or last node
			if (pNodeItem->m_pNode != gTransferFunction.m_Nodes.front() && pNodeItem->m_pNode != gTransferFunction.m_Nodes.back())
				gTransferFunction.RemoveNode(pNodeItem->m_pNode);
		}
	}
}