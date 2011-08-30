
#include "TransferFunctionCanvas.h"
#include "TransferFunction.h"
#include "NodeItem.h"

QTransferFunctionCanvas::QTransferFunctionCanvas(QGraphicsItem* pParent, QGraphicsScene* pGraphicsScene) :
	QGraphicsRectItem(pParent, pGraphicsScene),
	m_pBackgroundRectangle(NULL),
	m_BackgroundBrush(),
	m_BackgroundPen(),
	m_GridLinesHorizontal(),
	m_GridPenHorizontal(),
	m_GridPenVertical(),
	m_pPolygon(NULL),
	m_PolygonGradient(),
	m_pHistogram(NULL),
	m_CrossHairH(NULL),
	m_CrossHairV(NULL),
	m_RealisticsGradient(false),
	m_AllowUpdateNodes(true),
	m_NodeItems(),
	m_EdgeItems(),
	m_BackgroundZ(0),
	m_GridZ(100),
	m_HistogramZ(200),
	m_EdgeZ(300),
	m_PolygonZ(400),
	m_NodeZ(500),
	m_CrossHairZ(600)
{
	// Create polygon graphics item
	m_pPolygon = new QGraphicsPolygonItem;
	m_pPolygon->setParentItem(this);
	m_pPolygon->setPen(QPen(Qt::PenStyle::NoPen));

	// Histogram
	m_pHistogram = new QGraphicsPolygonItem;
	m_pHistogram->setParentItem(this);
	m_pHistogram->setBrush(QColor(200, 20, 20, 50));
	m_pHistogram->setPen(QPen(QBrush(QColor(100, 10, 10, 150)), 0.5f));

	// Horizontal cross hair
	m_CrossHairH = new QGraphicsLineItem(this);
	m_CrossHairH->setPen(QPen(Qt::darkGray, 0.2f));
	m_CrossHairH->setZValue(m_CrossHairZ);
	m_CrossHairH->setVisible(false);

	// Horizontal cross hair
	m_CrossHairV = new QGraphicsLineItem(this);
	m_CrossHairV->setPen(QPen(Qt::darkGray, 0.2f));
	m_CrossHairV->setZValue(m_CrossHairZ);
	m_CrossHairV->setVisible(false);

	// Cross hair text
	m_CrossHairText = new QGraphicsTextItem(this);
	m_CrossHairText->setZValue(m_CrossHairZ);
	m_CrossHairText->setVisible(false);

	// Background styling
	m_BackgroundBrush.setColor(QColor(Qt::gray));
	m_BackgroundBrush.setStyle(Qt::BrushStyle::SolidPattern);

	// Make sure the background rectangle is drawn behind everything else
	setZValue(m_BackgroundZ);

	m_pBackgroundRectangle = new QGraphicsRectItem(this);
	m_pBackgroundRectangle->setZValue(10);
	m_pBackgroundRectangle->setPen(Qt::NoPen);
	m_pBackgroundRectangle->setBrush(Qt::gray);

	// Grid
	QVector<qreal> dashes;
	dashes << 13 << 13;

	m_GridPenHorizontal.setColor(QColor(100, 100, 100, 100));
	m_GridPenHorizontal.setWidthF(0.6f);
	m_GridPenHorizontal.setStyle(Qt::PenStyle::DashLine);
	m_GridPenHorizontal.setDashPattern(dashes);

	m_GridPenVertical.setColor(QColor(100, 100, 100, 200));
	m_GridPenVertical.setWidthF(0.6f);
	m_GridPenVertical.setStyle(Qt::PenStyle::SolidLine);
//	m_GridPenVertical.setDashPattern(dashes);

	// Update the canvas
	Update();

	// We are going to catch hover events
	setAcceptHoverEvents(true);
}

void QTransferFunctionCanvas::hoverEnterEvent(QGraphicsSceneHoverEvent* pEvent)
{
	QGraphicsRectItem::hoverEnterEvent(pEvent);

	// Don't overwrite styling when selected
	if (isSelected())
		return;

	// Show crosshair
//	m_CrossHairH->setVisible(true);
//	m_CrossHairV->setVisible(true);
//	m_CrossHairText->setVisible(true);
}

void QTransferFunctionCanvas::hoverLeaveEvent(QGraphicsSceneHoverEvent* pEvent)
{
	QGraphicsRectItem::hoverLeaveEvent(pEvent);

	// Don't overwrite styling when selected
	if (isSelected())
		return;

	// Hide crosshair
//	m_CrossHairH->setVisible(false);
//	m_CrossHairV->setVisible(false);
//	m_CrossHairText->setVisible(false);
}

void QTransferFunctionCanvas::hoverMoveEvent(QGraphicsSceneHoverEvent* pEvent)
{
	QGraphicsRectItem::hoverMoveEvent(pEvent);

	// Don't overwrite styling when selected
	if (isSelected())
		return;

//	m_CrossHairH->setPos(QPointF(pEvent->pos().x(), 0));
//	m_CrossHairV->setPos(QPointF(0, pEvent->pos().y()));
//	m_CrossHairText->setPos(pEvent->pos());
//	m_CrossHairText->setPlainText("[" + QString::number(pEvent->pos().x()) + ", " + QString::number(pEvent->pos().y()) + "]");
}

void QTransferFunctionCanvas::Update(void)
{
	UpdateNodes();
	UpdateEdges();
	UpdateGradient();
	UpdatePolygon();
	UpdateCrossHairs();

	m_pBackgroundRectangle->setRect(rect());
}

void QTransferFunctionCanvas::UpdateGrid(void)
{
	// Horizontal grid lines
	const float DeltaY = 0.2f * rect().height();

	// Remove old horizontal grid lines
	foreach(QGraphicsLineItem* pLine, m_GridLinesHorizontal)
		scene()->removeItem(pLine);

	// Clear the edges list
	m_GridLinesHorizontal.clear();

	for (int i = 1; i < 5; i++)
	{
		// Create a new grid line
		QGraphicsLineItem* pLine = new QGraphicsLineItem(QLineF(0, i * DeltaY, rect().width(), i * DeltaY));

		pLine->setPen(m_GridPenHorizontal);

		// Depth ordering
		pLine->setZValue(m_GridZ);

		// Set parent
		pLine->setParentItem(this);

		// Add it to the list so we can remove them from the canvas when needed
		m_GridLinesHorizontal.append(pLine);
	}

	float GridInterval = 50.0f;

	int Num = ceilf(rect().width() / GridInterval);

	for (int i = 0; i < Num; i++)
	{
		// Create a new grid line
		QGraphicsLineItem* pLine = new QGraphicsLineItem(QLineF(i * GridInterval, 0.0f, i * GridInterval, rect().height()));

		pLine->setPen(m_GridPenVertical);

		// Depth ordering
		pLine->setZValue(m_GridZ);

		// Set parent
		pLine->setParentItem(this);

		// Add it to the list so we can remove them from the canvas when needed
		m_GridLinesHorizontal.append(pLine);
	}
}

void QTransferFunctionCanvas::UpdateHistogram(void)
{
	QPolygonF Polygon;

	// Set the gradient stops
	for (int i = 0; i < gTransferFunction.GetHistogram().m_Bins.size(); i++)
	{
		// Compute polygon point in scene coordinates
		QPointF ScenePoint = TransferFunctionToScene(QPointF(i, logf((float)gTransferFunction.GetHistogram().m_Bins[i]) / logf(1.5f * (float)gTransferFunction.GetHistogram().m_Max)));

		if (i == 0)
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(rect().height());

			Polygon.append(CenterCopy);
		}

		Polygon.append(ScenePoint);

		if (i == (gTransferFunction.GetHistogram().m_Bins.size() - 1))
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(rect().height());

			Polygon.append(CenterCopy);
		}
	}

	// Depth order
	m_pHistogram->setZValue(m_HistogramZ);

	// Update the polygon geometry
	m_pHistogram->setPolygon(Polygon);
}

void QTransferFunctionCanvas::UpdateNodes(void)
{
	if (!m_AllowUpdateNodes)
		return;

	// Remove old nodes
	foreach(QNodeItem* pNodeItem, m_NodeItems)
		scene()->removeItem(pNodeItem);

	// Clear the node items list
	m_NodeItems.clear();

	for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
	{
		QNode& Node = gTransferFunction.GetNode(i);

		// Create new node item
		QNodeItem* pNodeItem = new QNodeItem(NULL, &Node, this);

		// Make it child of us
		pNodeItem->setParentItem(this);

		// Compute node center in canvas coordinates
		QPointF NodeCenter = TransferFunctionToScene(QPointF(Node.GetIntensity(), Node.GetOpacity()));

		pNodeItem->m_SuspendUpdate = true;

		// Set node position
		pNodeItem->setPos(NodeCenter);

		pNodeItem->m_SuspendUpdate = false;

		// Make sure node items are rendered on top
		pNodeItem->setZValue(m_NodeZ);

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
		pLine->setPen(QPen(QColor(240, 160, 30), 1.2f));

		// Ensure the item is drawn in the right order
		pLine->setZValue(m_EdgeZ);

		m_EdgeItems.append(pLine);
	}
}

void QTransferFunctionCanvas::UpdateGradient(void)
{
	if (m_RealisticsGradient)
	{
		m_PolygonGradient.setStart(0, 0);
		m_PolygonGradient.setFinalStop(rect().right(), rect().top());

		QGradientStops GradientStops;

		for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
		{
			const QNode& Node = gTransferFunction.GetNode(i);

			QColor Color = Node.GetColor();

			// Clamp node opacity to obtain valid alpha for display
			float Alpha = qMin(1.0f, qMax(0.0f, Node.GetOpacity()));

			Color.setAlphaF(0.5f * Alpha);

			// Add a new gradient stop
			GradientStops.append(QGradientStop(Node.GetNormalizedIntensity(), Color));	
		}

		m_PolygonGradient.setStops(GradientStops);
	}
	else
	{
		m_PolygonGradient.setStart(0, rect().bottom());
		m_PolygonGradient.setFinalStop(0, rect().top());

		QGradientStops GradientStops;

		GradientStops.append(QGradientStop(0, QColor(230, 230, 230, 0)));
		GradientStops.append(QGradientStop(1, QColor(230, 230, 230, 220)));

		m_PolygonGradient.setStops(GradientStops);
	}
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

	// Depth order
	m_pPolygon->setZValue(m_PolygonZ);

	// Update the polygon geometry
	m_pPolygon->setPolygon(Polygon);

	// Give the polygon a gradient brush
	m_pPolygon->setBrush(QBrush(m_PolygonGradient));
}

void QTransferFunctionCanvas::UpdateCrossHairs(void)
{
	// Horizontal cross hair
	m_CrossHairH->setLine(QLineF(rect().topLeft(), rect().bottomLeft()));

	// Vertical cross hair
	m_CrossHairV->setLine(QLineF(rect().topLeft(), rect().topRight()));
}

// Maps from scene coordinates to transfer function coordinates
QPointF QTransferFunctionCanvas::SceneToTransferFunction(const QPointF& ScenePoint)
{
	const float NormalizedX = ScenePoint.x() / (float)rect().width();
	const float NormalizedY = 1.0f - (ScenePoint.y() / (float)rect().height());

	const float TfX = gTransferFunction.GetRangeMin() + NormalizedX * gTransferFunction.GetRange();
	const float TfY = NormalizedY;

	return QPointF(TfX, TfY);
}

// Maps from transfer function coordinates to scene coordinates
QPointF QTransferFunctionCanvas::TransferFunctionToScene(const QPointF& TfPoint)
{
	const float NormalizedX = (TfPoint.x() - gTransferFunction.GetRangeMin()) / gTransferFunction.GetRange();
	const float NormalizedY = 1.0f - TfPoint.y();

	const float SceneX = NormalizedX * rect().width();
	const float SceneY = NormalizedY * rect().height();

	return QPointF(SceneX, SceneY);
}