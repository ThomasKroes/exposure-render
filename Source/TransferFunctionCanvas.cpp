
#include "TransferFunctionCanvas.h"
#include "TransferFunction.h"
#include "NodeItem.h"

QTransferFunctionCanvas::QTransferFunctionCanvas(QGraphicsItem* pParent, QGraphicsScene* pGraphicsScene) :
	QGraphicsRectItem(pParent, pGraphicsScene),
	m_BackgroundBrush(),
	m_BackgroundPen(),
	m_GridLinesHorizontal(),
	m_GridPenHorizontal(),
	m_GridPenVertical(),
	m_pPolygon(NULL),
	m_PolygonGradient(),
	m_pHistogram(NULL),
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

	// Make sure the background rectangle is drawn behind everything else
	setZValue(m_BackgroundZ);

	// Grid
	m_GridPenHorizontal.setColor(QColor(100, 100, 100, 200));
	m_GridPenHorizontal.setWidthF(0.5f);
	m_GridPenVertical.setColor(QColor(100, 100, 100, 200));
	m_GridPenVertical.setWidthF(0.5f);

	// Update the canvas
	Update();
}

void QTransferFunctionCanvas::Update(void)
{
	UpdateGrid();
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
	for (int i = 0; i < gTransferFunction.m_Histogram.m_Bins.size(); i++)
	{
		// Compute polygon point in scene coordinates
		QPointF ScenePoint = TransferFunctionToScene(QPointF(i, logf((float)gTransferFunction.m_Histogram.m_Bins[i]) / logf(1.5f * (float)gTransferFunction.m_Histogram.m_Max)));

		if (i == 0)
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(rect().height());

			Polygon.append(CenterCopy);
		}

		Polygon.append(ScenePoint);

		if (i == (gTransferFunction.m_Histogram.m_Bins.size() - 1))
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

	// Create the node items
	foreach(QNode* pNode, gTransferFunction.m_Nodes)
	{
		// Create new node item
		QNodeItem* pNodeItem = new QNodeItem(NULL, pNode, this);

		// Make it child of us
		pNodeItem->setParentItem(this);

		// Compute node center in canvas coordinates
		QPointF NodeCenter = TransferFunctionToScene(QPointF(pNode->GetX(), pNode->GetY()));

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
		pLine->setPen(QPen(QColor(110, 110, 100), 0.5));

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

		// Set the gradient stops
		foreach(QNode* pNode, gTransferFunction.m_Nodes)
		{
			QColor Color = pNode->GetColor();

			// Clamp node opacity to obtain valid alpha for display
			float Alpha = qMin(1.0f, qMax(0.0f, pNode->GetOpacity()));

			Color.setAlphaF(0.1f);

			// Add a new gradient stop
			GradientStops.append(QGradientStop(pNode->GetNormalizedX(), Color));
		}

		m_PolygonGradient.setStops(GradientStops);
	}
	else
	{
		m_PolygonGradient.setStart(0, rect().bottom());
		m_PolygonGradient.setFinalStop(0, rect().top());

		QGradientStops GradientStops;

		GradientStops.append(QGradientStop(0, QColor(255, 255, 255, 0)));
		GradientStops.append(QGradientStop(1, QColor(255, 255, 255, 255)));

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