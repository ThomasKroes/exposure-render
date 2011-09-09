
#include "TransferFunctionCanvas.h"
#include "TransferFunction.h"
#include "NodeItem.h"

QGridLine::QGridLine(QTransferFunctionCanvas* pTransferFunctionCanvas) :
	QGraphicsLineItem(pTransferFunctionCanvas),
	m_pTransferFunctionCanvas(pTransferFunctionCanvas)
{
}

QTransferFunctionCanvas::QTransferFunctionCanvas(QGraphicsItem* pParent, QGraphicsScene* pGraphicsScene) :
	QGraphicsRectItem(pParent, pGraphicsScene),
	m_BackgroundRectangle(),
	m_BackgroundBrush(),
	m_BackgroundPen(),
	m_GridLines(),
	m_GridPenHorizontal(),
	m_GridPenVertical(),
	m_Polygon(),
	m_PolygonGradient(),
	m_Histogram(),
	m_RealisticsGradient(false),
	m_AllowUpdateNodes(true),
	m_Nodes(),
	m_Edges(),
	m_BackgroundZ(0),
	m_GridZ(100),
	m_HistogramZ(200),
	m_EdgeZ(300),
	m_PolygonZ(400),
	m_NodeZ(500),
	m_CrossHairZ(600)
{
	// Create polygon graphics item
	m_Polygon.setParentItem(this);
	m_Polygon.setPen(QPen(Qt::NoPen));

	// Histogram
	m_Histogram.setParentItem(this);
	m_Histogram.setPen(QPen(QBrush(QColor::fromHsl(0, 100, 110)), 0.65f));

	// Background styling
	m_BackgroundBrush.setColor(QColor(Qt::gray));
	m_BackgroundBrush.setStyle(Qt::SolidPattern);

	// Make sure the background rectangle is drawn behind everything else
	setZValue(m_BackgroundZ);

	m_BackgroundRectangle.setParentItem(this);
	m_BackgroundRectangle.setZValue(10);
	m_BackgroundRectangle.setPen(Qt::NoPen);
	m_BackgroundRectangle.setBrush(Qt::gray);

	// Grid
	QVector<qreal> dashes;
	dashes << 13 << 13;

	m_GridPenHorizontal.setColor(QColor(100, 100, 100, 100));
	m_GridPenHorizontal.setWidthF(0.6f);
	m_GridPenHorizontal.setStyle(Qt::DashLine);
	m_GridPenHorizontal.setDashPattern(dashes);

	m_GridPenVertical.setColor(QColor(100, 100, 100, 200));
	m_GridPenVertical.setWidthF(0.6f);
	m_GridPenVertical.setStyle(Qt::SolidLine);
//	m_GridPenVertical.setDashPattern(dashes);

	// Update the canvas
	Update();

	// We are going to catch hover events
	setAcceptHoverEvents(true);

	setEnabled(false);
}

QTransferFunctionCanvas::~QTransferFunctionCanvas(void)
{
	for (int i = 0; i < m_GridLines.size(); i++)
		scene()->removeItem(m_GridLines[i]);

	for (int i = 0; i < m_Nodes.size(); i++)
		scene()->removeItem(m_Nodes[i]);

	for (int i = 0; i < m_Edges.size(); i++)
		scene()->removeItem(m_Edges[i]);
}

void QTransferFunctionCanvas::Update(void)
{
	UpdateNodes();
	UpdateEdges();
	UpdateGradient();
	UpdatePolygon();

	m_BackgroundRectangle.setRect(rect());
}

void QTransferFunctionCanvas::UpdateGrid(void)
{
	for (int i = 0; i < m_GridLines.size(); i++)
		scene()->removeItem(m_GridLines[i]);

	// Clear the edges list
	m_GridLines.clear();

	// Horizontal grid lines
	const float DeltaY = 0.1f * rect().height();

	for (int i = 1; i < 10; i++)
	{
		QGridLine* pGridLine = new QGridLine(this);
		m_GridLines.append(pGridLine);

		pGridLine->setLine(QLineF(0, i * DeltaY, rect().width(), i * DeltaY));
		pGridLine->setPen(m_GridPenHorizontal);
		pGridLine->setZValue(m_GridZ);
	}

	float GridInterval = 50.0f;

	int Num = ceilf(rect().width() / GridInterval);

	for (int i = 0; i < Num; i++)
	{
		QGridLine* pGridLine = new QGridLine(this);
		m_GridLines.append(pGridLine);

		pGridLine->setLine(QLineF(i * GridInterval, 0.0f, i * GridInterval, rect().height()));
		pGridLine->setPen(m_GridPenVertical);
		pGridLine->setZValue(m_GridZ);
	}
}

void QTransferFunctionCanvas::UpdateHistogram(void)
{
	m_Histogram.setVisible(gHistogram.GetEnabled());

	if (!gHistogram.GetEnabled())
		return;

	QPolygonF Polygon;

	QLinearGradient LinearGradient;

	LinearGradient.setStart(0, rect().bottom());
	LinearGradient.setFinalStop(0, rect().top());

	QGradientStops GradientStops;

	GradientStops.append(QGradientStop(0, QColor::fromHsl(0, 100, 150, 0)));
	GradientStops.append(QGradientStop(1, QColor::fromHsl(0, 100, 150, 255)));

	LinearGradient.setStops(GradientStops);

	// Set the gradient stops
	for (int i = 0; i < gHistogram.GetBins().size(); i++)
	{
		// Compute polygon point in scene coordinates
		QPointF ScenePoint = TransferFunctionToScene(QPointF(i, logf((float)gHistogram.GetBins()[i]) / logf(1.5f * (float)gHistogram.GetMax())));

		if (i == 0)
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(rect().height());

			Polygon.append(CenterCopy);
		}

		Polygon.append(ScenePoint);

		if (i == (gHistogram.GetBins().size() - 1))
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(rect().height());

			Polygon.append(CenterCopy);
		}
	}

	// Depth order
	m_Histogram.setZValue(m_HistogramZ);

	// Update the polygon geometry
	m_Histogram.setPolygon(Polygon);
	m_Histogram.setBrush(QBrush(LinearGradient));
}

void QTransferFunctionCanvas::UpdateNodes(void)
{
	if (!m_AllowUpdateNodes)
		return;

	for (int i = 0; i < m_Nodes.size(); i++)
		scene()->removeItem(m_Nodes[i]);

	// Clear the node items list
	m_Nodes.clear();

	// Reserve
	m_Nodes.reserve(gTransferFunction.GetNodes().size());
	
	for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
	{
		QNode& Node = gTransferFunction.GetNode(i);

		QNodeItem* pNodeItem = new QNodeItem(this, &Node);
		m_Nodes.append(pNodeItem);

		// Compute node center in canvas coordinates
		QPointF NodeCenter = TransferFunctionToScene(QPointF(Node.GetIntensity(), Node.GetOpacity()));

		pNodeItem->setZValue(m_NodeZ);
		pNodeItem->m_SuspendUpdate = true;
		pNodeItem->setPos(NodeCenter);
		pNodeItem->m_SuspendUpdate = false;
	}

	if (gTransferFunction.GetSelectedNode())
	{
		for (int i = 0; i < m_Nodes.size(); i++)
		{
			if (m_Nodes[i]->m_pNode->GetID() == gTransferFunction.GetSelectedNode()->GetID())
				m_Nodes[i]->setSelected(true);
		}
	}
}

void QTransferFunctionCanvas::UpdateEdges(void)
{
	for (int i = 0; i < m_Edges.size(); i++)
		scene()->removeItem(m_Edges[i]);

	// Clear the edges list
	m_Edges.clear();

	for (int i = 1; i < m_Nodes.size(); i++)
	{
		QPointF PtFrom(m_Nodes[i - 1]->pos());
		QPointF PtTo(m_Nodes[i]->pos());

		QEdgeItem* pEdge = new QEdgeItem(this);
		m_Edges.append(pEdge);

		pEdge->setLine(QLineF(PtFrom, PtTo));		
		pEdge->setZValue(m_EdgeZ);
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

			QColor Color = Node.GetDiffuse();

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

		GradientStops.append(QGradientStop(0, QColor::fromHsl(0, 30, 150, 30)));
		GradientStops.append(QGradientStop(1, QColor::fromHsl(0, 30, 150, 150)));

		m_PolygonGradient.setStops(GradientStops);
	}
}

void QTransferFunctionCanvas::UpdatePolygon(void)
{
	QPolygonF Polygon;

	// Set the gradient stops
	for (int i = 0; i < m_Nodes.size(); i++)
	{
		QNodeItem* pNodeItem = m_Nodes[i];

		// Compute polygon point in scene coordinates
		QPointF ScenePoint = pNodeItem->pos();

		if (i == 0)
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(rect().height());

			Polygon.append(CenterCopy);
		}

		Polygon.append(ScenePoint);

		if (i == m_Nodes.size() - 1)
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(rect().height());

			Polygon.append(CenterCopy);
		}
	}

	// Depth order
	m_Polygon.setZValue(m_PolygonZ);

	// Update the polygon geometry
	m_Polygon.setPolygon(Polygon);

	// Give the polygon a gradient brush
	m_Polygon.setBrush(QBrush(m_PolygonGradient));
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




