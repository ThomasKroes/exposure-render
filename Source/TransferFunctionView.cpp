
#include "TransferFunctionView.h"
#include "TransferFunction.h"
#include "NodeItem.h"

// Compare two nodes
bool CompareNodes(QNode* pNodeA, QNode* pNodeB)
{
	return pNodeA->GetPosition() < pNodeB->GetPosition();
}

// Compare two node items
bool CompareNodeItems(QNodeItem* pNodeItemA, QNodeItem* pNodeItemB)
{
	return pNodeItemA->m_pNode->GetPosition() < pNodeItemB->m_pNode->GetPosition();
}

QTransferFunctionCanvas::QTransferFunctionCanvas(QGraphicsItem* pParent) :
	QGraphicsRectItem(pParent)
{
	// Add polygon graphics item
	m_pPolygon = new QGraphicsPolygonItem;

	// Ensure the item is drawn in the right order
	m_pPolygon->setZValue(500);
}

void QTransferFunctionCanvas::mousePressEvent(QGraphicsSceneMouseEvent* pGraphicsSceneMouseEvent)
{
	QGraphicsRectItem::mousePressEvent(pGraphicsSceneMouseEvent);

	// Get node item under cursor
	QNodeItem* pNodeItem = dynamic_cast<QNodeItem*>(scene()->itemAt(pGraphicsSceneMouseEvent->pos()));

	if (!pNodeItem)
	{
		// Add a new node if the user clicked the left button
		if (pGraphicsSceneMouseEvent->button() == Qt::MouseButton::LeftButton)
		{
			/*
			QPointF TfPoint = SceneToTransferFunction(m_pCanvas->mapFromScene(pEvent->posF()));

			int R = (int)(((float)rand() / (float)RAND_MAX) * 255.0f);
			int G = (int)(((float)rand() / (float)RAND_MAX) * 255.0f);
			int B = (int)(((float)rand() / (float)RAND_MAX) * 255.0f);

			// Create new transfer function node
			QNode* pNode = new QNode(m_pTransferFunction, TfPoint.x(), TfPoint.y(), QColor(R, G, B, 255));

			// Add to node list
			gTransferFunction.AddNode(pNode);

			// Select it immediately
			SetSelectedNode(pNode);
			*/
		}
		else
		{
			// Other wise no node selection
			SetSelectedNode(NULL);
		}
	}
	else
	{
		SetSelectedNode(pNodeItem->m_pNode);
	}
}

void QTransferFunctionCanvas::SetSelectedNode(QNode* pSelectedNode)
{
	gTransferFunction.SetSelectedNode(pSelectedNode);
}

void QTransferFunctionCanvas::Update(void)
{
	UpdateHistogram();
	UpdateNodes();
	UpdateNodeRanges();
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
		QGraphicsLineItem* pLine = new QGraphicsLineItem(QLineF(QPointF(rect().left(), rect().top() + i * DeltaY), QPointF(rect().right(), rect().top() + i * DeltaY)));

		pLine->setPen(m_GridPenHorizontal);

		scene()->addItem(pLine);
		m_GridLinesHorizontal.append(pLine);
	}
}

void QTransferFunctionCanvas::UpdateHistogram(void)
{
}

void QTransferFunctionCanvas::UpdateEdges(void)
{
	// Remove old edges
	foreach(QGraphicsLineItem* pLine, m_Edges)
		scene()->removeItem(pLine);

	// Clear the edges list
	m_Edges.clear();

	for (int i = 1; i < m_Nodes.size(); i++)
	{
		QPointF PtFrom(m_Nodes[i - 1]->rect().center());
		QPointF PtTo(m_Nodes[i]->rect().center());

		QGraphicsLineItem* pLine = new QGraphicsLineItem(QLineF(PtFrom, PtTo));
		
		// Set the pen
		pLine->setPen(QPen(QColor(110, 110, 100), 0.5));

		// Ensure the item is drawn in the right order
		pLine->setZValue(800);

		m_Edges.append(pLine);
		scene()->addItem(pLine);
	}
}

void QTransferFunctionCanvas::UpdateNodes(void)
{
	// Set the gradient stops
	foreach(QNodeItem* pNodeGraphics, m_Nodes)
	{
		QPointF SceneCenter = TransferFunctionToScene(QPointF(pNodeGraphics->m_pNode->GetX(), pNodeGraphics->m_pNode->GetY()));

//		Center.setX(rect().left() + rect().width() * pNodeGraphics->m_pNode->GetNormalizedX());
//		Center.setY(rect().top() + rect().height() - (pNodeGraphics->m_pNode->GetOpacity() * rect().height()));
		
		pNodeGraphics->SetCenter(SceneCenter);
	}
}

void QTransferFunctionCanvas::UpdateNodeRanges(void)
{
	if (gTransferFunction.m_Nodes.size() < 2)
		return;

	for (int i = 0; i < gTransferFunction.m_Nodes.size(); i++)
	{
		QNode* pNode = gTransferFunction.m_Nodes[i];

		if (pNode == gTransferFunction.m_Nodes.front())
		{
			pNode->m_MinX = 0.0f;
			pNode->m_MaxX = 0.0f;
		}
		else if (pNode == gTransferFunction.m_Nodes.back())
		{
			pNode->m_MinX = gTransferFunction.m_RangeMax;
			pNode->m_MaxX = gTransferFunction.m_RangeMax;
		}
		else
		{
			QNode* pNodeLeft	= gTransferFunction.m_Nodes[i - 1];
			QNode* pNodeRight	= gTransferFunction.m_Nodes[i + 1];

			pNode->m_MinX = pNodeLeft->GetPosition();
			pNode->m_MaxX = pNodeRight->GetPosition();
		}
	}
}

void QTransferFunctionCanvas::UpdateGradient(void)
{
	m_LinearGradient.setStart(rect().left(), rect().top());
	m_LinearGradient.setFinalStop(rect().right(), rect().top());

	QGradientStops GradientStops;

	// Set the gradient stops
	foreach(QNode* pNode, gTransferFunction.m_Nodes)
	{
		QColor Color = pNode->GetColor();

		// Clamp node opacity to obtain valid alpha for display
		float Alpha = qMin(1.0f, qMax(0.0f, pNode->GetOpacity()));

		Color.setAlphaF(0.2f);

		// Add a new gradient stop
		GradientStops.append(QGradientStop(pNode->GetNormalizedX(), Color));
	}

	m_LinearGradient.setStops(GradientStops);
}

void QTransferFunctionCanvas::UpdatePolygon(void)
{
	QPolygonF Polygon;

	// Set the gradient stops
	for (int i = 0; i < m_Nodes.size(); i++)
	{
		QNodeItem* pNodeGraphics = m_Nodes[i];

		// Compute polygon point in scene coordinates
		QPointF ScenePoint = TransferFunctionToScene(QPointF(pNodeGraphics->m_pNode->GetX(), pNodeGraphics->m_pNode->GetY()));

		if (pNodeGraphics == m_Nodes.front())
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(rect().bottom());

			Polygon.append(CenterCopy);
		}

		Polygon.append(ScenePoint);

		if (pNodeGraphics == m_Nodes.back())
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(rect().bottom());

			Polygon.append(CenterCopy);
		}
	}

	m_pPolygon->setPolygon(Polygon);
	m_pPolygon->setBrush(QBrush(m_LinearGradient));
	m_pPolygon->setPen(QPen(Qt::PenStyle::NoPen));
}

// Maps from scene coordinates to transfer function coordinates
QPointF QTransferFunctionCanvas::SceneToTransferFunction(const QPointF& ScenePoint)
{
	const float NormalizedX = (ScenePoint.x() - (float)rect().left()) / (float)rect().width();
	const float NormalizedY = 1.0f - ((ScenePoint.y() - (float)rect().top()) / (float)rect().height());

	const float TfX = gTransferFunction.m_RangeMin + NormalizedX * gTransferFunction.m_Range;
	const float TfY = NormalizedY;

	return QPointF(TfX, TfY);
}

// Maps from transfer function coordinates to scene coordinates
QPointF QTransferFunctionCanvas::TransferFunctionToScene(const QPointF& TfPoint)
{
	const float NormalizedX = (TfPoint.x() - gTransferFunction.m_RangeMin) / gTransferFunction.m_Range;
	const float NormalizedY = 1.0f - TfPoint.y();

	const float SceneX = rect().left() + NormalizedX * rect().width();
	const float SceneY = rect().top() + NormalizedY * rect().height();

	return QPointF(SceneX, SceneY);
}

QTransferFunctionView::QTransferFunctionView(QWidget* pParent) :
	QGraphicsView(pParent),
	m_pGraphicsScene(NULL),
	m_pTransferFunctionCanvas(NULL),
	m_Margin(4.0f)
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
	setScene(scene());

	// Turn antialiasing on
	setRenderHint(QPainter::Antialiasing);

	// Respond to changes in the transfer function
	connect(&gTransferFunction, SIGNAL(FunctionChanged()), this, SLOT(Update()));

	/*
	connect(m_pTransferFunction, SIGNAL(NodeAdd(QNode*)), this, SLOT(OnNodeAdd(QNode*)));

	// Respond to changes in node selection
	connect(m_pTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));

	

	// Add the polygon
	scene()->addItem(m_pPolygon);

	
	// Create rectangle for outline
	m_pOutline = new QGraphicsRectItem;
	m_pOutline->setRect(rect());
	m_pOutline->setBrush(QBrush(Qt::BrushStyle::NoBrush));
	m_pOutline->setPen(QPen(Qt::darkGray));

	// Ensure the item is drawn in the right order
	m_pOutline->setZValue(1000);
	

	// Add the rectangle
//	scene()->addItem(m_pOutline);

	// Create canvas
	m_pCanvas = new QGraphicsRectItem;
	m_pCanvas->setRect(rect());
	m_pCanvas->setBrush(QBrush(QColor(190, 190, 190)));
	m_pCanvas->setPen(QPen(Qt::darkGray));

	// Add the rectangle
	scene()->addItem(m_pCanvas);

	// Render
	Update();
	UpdateNodes();

	// Configure pen
	m_GridPenHorizontal.setStyle(Qt::PenStyle::DashLine);
	m_GridPenHorizontal.setColor(QColor(75, 75, 75, 120));
//	m_GridPenHorizontal.setWidthF(1.0f);

	m_Text = new QGraphicsTextItem();
	m_Text->setPos(rect().bottomLeft());
	m_Text->setTextWidth(rect().width());
	m_Text->setHtml("<center>Density</center>");
*/
//	scene()->addItem(m_Text);
}

void QTransferFunctionView::drawBackground(QPainter* pPainter, const QRectF& Rectangle)
{
	// Base class
	QGraphicsView::drawBackground(pPainter, Rectangle);
	
	setBackgroundBrush(QBrush(QColor(240, 240, 240)));
}

void QTransferFunctionView::OnNodeAdd(QNode* pTransferFunctionNode)
{
	/*
	QNodeItem* pNodeItem = new QNodeItem(NULL, pTransferFunctionNode, this);
	
	// Ensure the item is drawn in the right order
	pNodeItem->setZValue(900);

	scene()->addItem(pNodeItem);

	m_Nodes.append(pNodeItem);
	
	qSort(m_Nodes.begin(), m_Nodes.end(), CompareNodeItems);
	qSort(gTransferFunction.m_Nodes.begin(), gTransferFunction.m_Nodes.end(), CompareNodes);

	UpdateNodes();
	*/

}

void QTransferFunctionView::OnNodeSelectionChanged(QNode* pNode)
{
	/*
	if (pNode)
	{
		// Deselect all nodes
		foreach (QNodeItem* pNode, m_Nodes)
			pNode->setSelected(false);

		// Obtain node index
		const int NodeIndex = gTransferFunction.GetNodeIndex(pNode);

		// Select the node
		if (NodeIndex >= 0)
		{
			m_Nodes[NodeIndex]->setSelected(true);
			m_Nodes[NodeIndex]->update();
		}
	}
	else
	{
	}
	*/
}