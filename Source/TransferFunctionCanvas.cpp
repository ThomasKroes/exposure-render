
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
	m_Polygon.setParentItem(this);
	m_Polygon.setPen(QPen(Qt::PenStyle::NoPen));

	// Histogram
	m_Histogram.setParentItem(this);
	m_Histogram.setBrush(QColor(200, 20, 20, 50));
	m_Histogram.setPen(QPen(QBrush(QColor(100, 10, 10, 150)), 0.5f));

	// Background styling
	m_BackgroundBrush.setColor(QColor(Qt::gray));
	m_BackgroundBrush.setStyle(Qt::BrushStyle::SolidPattern);

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
	// Horizontal grid lines
	const float DeltaY = 0.2f * rect().height();

	// Clear the edges list
	m_GridLines.clear();

	for (int i = 1; i < 5; i++)
	{
		m_GridLines.append(QGridLine(this));

		QGridLine& GridLine = m_GridLines.back();

		GridLine.setLine(QLineF(0, i * DeltaY, rect().width(), i * DeltaY));
		GridLine.setPen(m_GridPenHorizontal);
		GridLine.setZValue(m_GridZ);
		GridLine.setParentItem(this);
	}

	float GridInterval = 50.0f;

	int Num = ceilf(rect().width() / GridInterval);

	for (int i = 0; i < Num; i++)
	{
		m_GridLines.append(QGridLine(this));

		QGridLine& GridLine = m_GridLines.back();

		GridLine.setLine(QLineF(i * GridInterval, 0.0f, i * GridInterval, rect().height()));
		GridLine.setPen(m_GridPenVertical);
		GridLine.setZValue(m_GridZ);
		GridLine.setParentItem(this);
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
	m_Histogram.setZValue(m_HistogramZ);

	// Update the polygon geometry
	m_Histogram.setPolygon(Polygon);
}

void QTransferFunctionCanvas::UpdateNodes(void)
{
	if (!m_AllowUpdateNodes)
		return;

	// Clear the node items list
	m_NodeItems.clear();

	for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
	{
		QNode& Node = gTransferFunction.GetNode(i);

		m_NodeItems.append(QNodeItem(this, &Node));

		QNodeItem& NodeItem = m_NodeItems.back();
		
		// Compute node center in canvas coordinates
		QPointF NodeCenter = TransferFunctionToScene(QPointF(Node.GetIntensity(), Node.GetOpacity()));

		NodeItem.setZValue(m_NodeZ);
		NodeItem.setParentItem(this);

		NodeItem.m_SuspendUpdate = true;
		
		NodeItem.setPos(NodeCenter);

		NodeItem.m_SuspendUpdate = false;
	}
}

void QTransferFunctionCanvas::UpdateEdges(void)
{
	// Clear the edges list
	m_EdgeItems.clear();

	for (int i = 1; i < m_NodeItems.size(); i++)
	{
		QPointF PtFrom(m_NodeItems[i - 1].pos());
		QPointF PtTo(m_NodeItems[i].pos());

		m_EdgeItems.append(QEdgeItem(this));

		QEdgeItem& Line = m_EdgeItems.back();

		Line.setLine(QLineF(PtFrom, PtTo));		
		Line.setParentItem(this);
		Line.setPen(QPen(QColor(240, 160, 30), 1.2f));
		Line.setZValue(m_EdgeZ);
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
		QNodeItem& NodeItem = m_NodeItems[i];

		// Compute polygon point in scene coordinates
		QPointF ScenePoint = NodeItem.pos();

		if (i == 0)
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(rect().height());

			Polygon.append(CenterCopy);
		}

		Polygon.append(ScenePoint);

		if (i == m_NodeItems.size() - 1)
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


