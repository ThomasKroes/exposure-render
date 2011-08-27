
#include "GradientView.h"
#include "TransferFunction.h"
#include "NodeItem.h"

QGradientView::QGradientView(QWidget* pParent, QTransferFunction* pTransferFunction) :
	QGraphicsView(pParent),
	m_pGraphicsScene(NULL),
	m_pTransferFunction(pTransferFunction),
	m_CheckerSize(10, 10),
	m_pGradientRectangle(NULL),
	m_LinearGradient()
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
	connect(m_pTransferFunction, SIGNAL(FunctionChanged()), this, SLOT(Update()));
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
	
	// Set the gradient stops
	foreach(QNode* pNode, m_pTransferFunction->m_Nodes)
	{
		pPainter->setPen(QPen(Qt::PenStyle::SolidLine));

		pPainter->drawLine(QPointF(pNode->GetNormalizedX() * rect().width(), 0.0f), QPointF(pNode->GetNormalizedX() * rect().width(), 20.0f));
	}
}

void QGradientView::resizeEvent(QResizeEvent* pResizeEvent)
{
	Update();
}

void QGradientView::Update(void)
{
	setSceneRect(rect());

	m_LinearGradient.setStart(rect().left(), rect().top());
	m_LinearGradient.setFinalStop(rect().right(), rect().top());

	QGradientStops GradientStops;

	// Set the gradient stops
	foreach(QNode* pTransferFunctionNode, m_pTransferFunction->m_Nodes)
	{
		QColor Color = pTransferFunctionNode->GetColor();

		// Clamp node opacity to obtain valid alpha for display
		float Alpha = qMin(1.0f, qMax(0.0f, pTransferFunctionNode->GetOpacity()));

		Color.setAlphaF(Alpha);

		GradientStops.append(QGradientStop((pTransferFunctionNode->GetPosition() - m_pTransferFunction->m_RangeMin) / m_pTransferFunction->m_Range, Color));
	}

	m_LinearGradient.setStops(GradientStops);

	m_pGradientRectangle->setRect(rect());
	m_pGradientRectangle->setBrush(QBrush(m_LinearGradient));
	m_pGradientRectangle->setPen(QPen(Qt::darkGray));
//	m_pGradientRectangle->setVisible(false);
}

void QGradientView::OnNodeAdd(QNode* pTransferFunctionNode)
{
}

void QGradientView::OnNodeRemove(QNode* pTransferFunctionNode)
{
}