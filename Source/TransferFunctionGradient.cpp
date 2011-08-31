
#include "TransferFunctionGradient.h"
#include "TransferFunction.h"
#include "NodeItem.h"

QTransferFunctionGradient::QTransferFunctionGradient(QGraphicsItem* pParent, QGraphicsScene* pGraphicsScene) :
	QGraphicsRectItem(pParent, pGraphicsScene),
	m_CheckerRectangles(),
	m_pGradientRectangle(NULL),
	m_LinearGradient()
{
	m_pGradientRectangle = new QGraphicsRectItem(this);
	m_pGradientRectangle->setPen(QPen(Qt::NoPen));
	m_pGradientRectangle->setZValue(100);
}

void QTransferFunctionGradient::Update(void)
{
	// Remove old horizontal grid lines
	foreach(QGraphicsRectItem* pRectangle, m_CheckerRectangles)
		scene()->removeItem(pRectangle);

	// Clear the edges list
	m_CheckerRectangles.clear();

	float m_CheckerSize = 0.5f * (float)rect().height();

	const int NumX = ceilf((float)rect().width() / m_CheckerSize);

	for (int i = 0; i < NumX; i++)
	{
		QGraphicsRectItem* pCheckerTop		= new QGraphicsRectItem(this);
		QGraphicsRectItem* pCheckerBottom	= new QGraphicsRectItem(this);

		const float Width = (i == NumX - 1) ? rect().width() - (NumX - 1) * m_CheckerSize : m_CheckerSize;

		pCheckerTop->setRect(i * m_CheckerSize, 0, Width, m_CheckerSize);
		pCheckerBottom->setRect(i * m_CheckerSize, m_CheckerSize, Width, m_CheckerSize);

		QBrush BrushTop(i % 2 == 0 ? Qt::gray : Qt::white);
		QBrush BrushBottom(i % 2 == 0 ? Qt::white : Qt::gray);

		// Depth ordering
		pCheckerTop->setZValue(0);
		pCheckerBottom->setZValue(0);

		// Set brush
		pCheckerTop->setBrush(BrushTop);
		pCheckerBottom->setBrush(BrushBottom);

		// Set pen
		pCheckerTop->setPen(Qt::NoPen);
		pCheckerBottom->setPen(Qt::NoPen);

		// Add to list
		m_CheckerRectangles.append(pCheckerTop);
		m_CheckerRectangles.append(pCheckerBottom);
	}

	m_LinearGradient.setStart(rect().left(), rect().top());
	m_LinearGradient.setFinalStop(rect().right(), rect().top());

	QGradientStops GradientStops;

	for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
	{
		QNode& Node = gTransferFunction.GetNode(i);

		// Get node color
		QColor Color = Node.GetColor();

		// Clamp node position to [0, 1]
		const float GradientStopPosition = qMin(1.0f, qMax(0.0f, (Node.GetIntensity() - gTransferFunction.GetRangeMin()) / gTransferFunction.GetRange()));

		// Clamp node opacity to [0, 1]
		const float GradientStopAlpha = qMin(1.0f, qMax(0.0f, Node.GetOpacity()));

		// Adjust color
		Color.setAlphaF(GradientStopAlpha);

		// Add the gradient stop
		GradientStops.append(QGradientStop(GradientStopPosition, Color));
	}

	m_LinearGradient.setStops(GradientStops);

	m_pGradientRectangle->setRect(rect());
	m_pGradientRectangle->setBrush(QBrush(m_LinearGradient));
//	m_pGradientRectangle->setPen(QPen(Qt::darkGray));
}

// Maps from scene coordinates to transfer function coordinates
QPointF QTransferFunctionGradient::SceneToTransferFunction(const QPointF& ScenePoint)
{
	const float NormalizedX = ScenePoint.x() / (float)rect().width();
	const float NormalizedY = 1.0f - (ScenePoint.y() / (float)rect().height());

	const float TfX = gTransferFunction.GetRangeMin() + NormalizedX * gTransferFunction.GetRange();
	const float TfY = NormalizedY;

	return QPointF(TfX, TfY);
}

// Maps from transfer function coordinates to scene coordinates
QPointF QTransferFunctionGradient::TransferFunctionToScene(const QPointF& TfPoint)
{
	const float NormalizedX = (TfPoint.x() - gTransferFunction.GetRangeMin()) / gTransferFunction.GetRange();
	const float NormalizedY = 1.0f - TfPoint.y();

	const float SceneX = NormalizedX * rect().width();
	const float SceneY = NormalizedY * rect().height();

	return QPointF(SceneX, SceneY);
}