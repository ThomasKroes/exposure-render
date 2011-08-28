
#include "GradientView.h"
#include "TransferFunction.h"
#include "NodeItem.h"

QGradientMarker::QGradientMarker(QGraphicsItem* pParent) :
	QGraphicsLineItem(pParent),
	m_pGradientRectangle(NULL),
	m_LinearGradient(),
	m_PolygonTop(NULL),
	m_PolygonBottom(NULL),
	m_PolygonSize(8, 4),
	m_Brush(QBrush(QColor(150, 150, 150, 255))),
	m_Pen(QBrush(QColor(90, 90, 90, 255)), 1.0f, Qt::PenStyle::SolidLine)
{
	// Create top polygon
	QPolygonF PolygonTop;
	PolygonTop.append(QPointF(-0.5f * m_PolygonSize.width(), 0.0f));
	PolygonTop.append(QPointF(0.0f, m_PolygonSize.height()));
	PolygonTop.append(QPointF(0.5f * m_PolygonSize.width(), 0.0f));

	m_PolygonTop = new QGraphicsPolygonItem(this);
	m_PolygonTop->setPolygon(PolygonTop);
	m_PolygonTop->setBrush(m_Brush);
	m_PolygonTop->setPen(m_Pen);

	// Create bottom polygon
	QPolygonF PolygonBottom;
	PolygonBottom.append(QPointF(-0.5f * m_PolygonSize.width(), pParent->boundingRect().bottom()));
	PolygonBottom.append(QPointF(0.0f, pParent->boundingRect().bottom() - m_PolygonSize.height()));
	PolygonBottom.append(QPointF(0.5f * m_PolygonSize.width(), pParent->boundingRect().bottom()));

	m_PolygonBottom = new QGraphicsPolygonItem(this);
	m_PolygonBottom->setPolygon(PolygonBottom);
	m_PolygonBottom->setBrush(m_Brush);
	m_PolygonBottom->setPen(m_Pen);

	setLine(QLineF(QPointF(0.0f, m_PolygonSize.height()), QPointF(0.0f, 20.0f - m_PolygonSize.height())));
	setPen(m_Pen);
	pen().setStyle(Qt::PenStyle::DashLine);
}

QGradientView::QGradientView(QWidget* pParent) :
	QGraphicsView(pParent),
	m_pGraphicsScene(NULL),
	m_CheckerSize(10, 10),
	m_pGradientRectangle(NULL),
	m_LinearGradient(),
	m_Markers()
{
	// Dimensions
	setFixedHeight(20);

	// Styling
	setFrameShadow(Sunken);
	setFrameShape(NoFrame);

	// Never show scrollbars
	setVerticalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);
	setHorizontalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);

	// Status and tooltip
	setStatusTip("Transfer function gradient");
	setToolTip("Transfer function gradient");

	// Create scene and apply
	m_pGraphicsScene = new QGraphicsScene(this);
	setScene(m_pGraphicsScene);

	// Turn antialiasing on
	setRenderHint(QPainter::Antialiasing);

	m_pGradientRectangle = new QGraphicsRectItem();
	m_pGradientRectangle->setBrush(QBrush(Qt::gray));
	m_pGradientRectangle->setPen(QPen(Qt::PenStyle::NoPen));
	m_pGraphicsScene->addItem(m_pGradientRectangle);

	// Setup connections
	connect(&gTransferFunction, SIGNAL(FunctionChanged()), this, SLOT(Update()));
}

void QGradientView::drawBackground(QPainter* pPainter, const QRectF& Rectangle)
{
	setSceneRect(rect());
	centerOn(Rectangle.center());

	const int NumX = ceilf(Rectangle.width() / m_CheckerSize.width());

	pPainter->setPen(QPen(Qt::PenStyle::NoPen));
	pPainter->setBrush(QBrush(Qt::gray));

	for (int i = 0; i < NumX; i++)
	{
		if (i % 2 == 0)
		{
			pPainter->drawRect(i * m_CheckerSize.width(), 0.0, m_CheckerSize.width(), m_CheckerSize.height());
		}
		else
		{
			pPainter->drawRect(i * m_CheckerSize.width(), m_CheckerSize.height(), m_CheckerSize.width(), m_CheckerSize.height());
		}
	}
}

void QGradientView::resizeEvent(QResizeEvent* pResizeEvent)
{
	Update();
}

void QGradientView::UpdateGradientMarkers(void)
{
	// Remove old gradient markers
	foreach(QGradientMarker* pMarker, m_Markers)
		m_pGraphicsScene->removeItem(pMarker);

	// Clear the marker list
	m_Markers.clear();

	// Create the gradient markers
	foreach(QNode* pNode, gTransferFunction.m_Nodes)
	{
		// New marker object
		QGradientMarker* pGradientMarker = new QGradientMarker(m_pGradientRectangle);
		
		// Position in scene coordinates, and add the marker to the scene
		QPointF ScenePointTop		= TfToScene(QPointF(pNode->GetPosition(), 0.0f));
		QPointF ScenePointBottom	= TfToScene(QPointF(pNode->GetPosition(), 1.0f));

		// Set the line and pen
		pGradientMarker->translate(ScenePointTop.x(), 0.0f);

		// Make sure the markers are drawn on top of the other geometry
		pGradientMarker->setZValue(1000);

		// Add the marker to the list
		m_Markers.append(pGradientMarker);
	}
}

void QGradientView::Update(void)
{
	setSceneRect(rect());

	m_LinearGradient.setStart(rect().left(), rect().top());
	m_LinearGradient.setFinalStop(rect().right(), rect().top());

	QGradientStops GradientStops;

	// Set the gradient stops
	foreach(QNode* pTransferFunctionNode, gTransferFunction.m_Nodes)
	{
		// Get node color
		QColor Color = pTransferFunctionNode->GetColor();

		// Clamp node position to [0, 1]
		const float GradientStopPosition = qMin(1.0f, qMax(0.0f, (pTransferFunctionNode->GetPosition() - gTransferFunction.m_RangeMin) / gTransferFunction.m_Range));

		// Clamp node opacity to [0, 1]
		const float GradientStopAlpha = qMin(1.0f, qMax(0.0f, pTransferFunctionNode->GetOpacity()));

		// Adjust color
		Color.setAlphaF(GradientStopAlpha);

		// Add the gradient stop
		GradientStops.append(QGradientStop(GradientStopPosition, Color));
	}

	m_LinearGradient.setStops(GradientStops);

	m_pGradientRectangle->setRect(rect());
	m_pGradientRectangle->setBrush(QBrush(m_LinearGradient));
	m_pGradientRectangle->setPen(QPen(Qt::darkGray));

	UpdateGradientMarkers();
}

QPointF QGradientView::SceneToTf(const QPointF& ScenePoint)
{
	const float NormalizedX = (ScenePoint.x() - (float)rect().left()) / (float)rect().width();
	const float NormalizedY = 1.0f - ((ScenePoint.y() - (float)rect().top()) / (float)rect().height());

	const float TfX = gTransferFunction.m_RangeMin + NormalizedX * gTransferFunction.m_Range;
	const float TfY = NormalizedY;

	return QPointF(TfX, TfY);
}

QPointF QGradientView::TfToScene(const QPointF& TfPoint)
{
	const float NormalizedX = (TfPoint.x() - gTransferFunction.m_RangeMin) / gTransferFunction.m_Range;
	const float NormalizedY = 1.0f - TfPoint.y();

	const float SceneX = rect().left() + NormalizedX * rect().width();
	const float SceneY = rect().top() + NormalizedY * rect().height();

	return QPointF(SceneX, SceneY);
}