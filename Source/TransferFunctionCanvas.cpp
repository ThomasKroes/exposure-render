
// Precompiled headers
#include "Stable.h"

#include "TransferFunctionCanvas.h"
#include "TransferFunction.h"
#include "NodeItem.h"

QTransferFunctionCanvas::QTransferFunctionCanvas(QGraphicsItem* pParent, QGraphicsScene* pGraphicsScene) :
	QGraphicsRectItem(pParent, pGraphicsScene),
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

	// Grid
	QVector<qreal> dashes;
	dashes << 13 << 13;


	// Update the canvas
	Update();

	// We are going to catch hover events
	setAcceptHoverEvents(true);

	setEnabled(false);
}

QTransferFunctionCanvas::~QTransferFunctionCanvas(void)
{
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
}

void QTransferFunctionCanvas::UpdateGrid(void)
{
	
}

void QTransferFunctionCanvas::UpdateNodes(void)
{
	/*
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
	*/
}

void QTransferFunctionCanvas::UpdateEdges(void)
{

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

	const float TfX = NormalizedX;
	const float TfY = NormalizedY;

	return QPointF(TfX, TfY);
}

// Maps from transfer function coordinates to scene coordinates
QPointF QTransferFunctionCanvas::TransferFunctionToScene(const QPointF& TfPoint)
{
	const float NormalizedX = TfPoint.x();
	const float NormalizedY = 1.0f - TfPoint.y();

	const float SceneX = NormalizedX * rect().width();
	const float SceneY = NormalizedY * rect().height();

	return QPointF(SceneX, SceneY);
}

void QTransferFunctionCanvas::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget /*= 0*/)
{
// 	if (gHistogram.GetPixMap())
// 		painter->drawPixmap(0, 0, 100, 100, *gHistogram.GetPixMap());
}
