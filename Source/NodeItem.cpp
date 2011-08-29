
#include "NodeItem.h"
#include "TransferFunction.h"
#include "TransferFunctionView.h"

float	QNodeItem::m_Radius				= 4.0f;
float	QNodeItem::m_RadiusHover		= 10.0f;
float	QNodeItem::m_RadiusSelected		= 10.0f;
QColor	QNodeItem::m_BackgroundColor	= QColor(230, 230, 230);
QColor	QNodeItem::m_TextColor			= QColor(20, 20, 20);
float	QNodeItem::m_PenWidth			= 1.7f;
float	QNodeItem::m_PenWidthHover		= 1.7f;
float	QNodeItem::m_PenWidthSelected	= 1.7f;
QColor	QNodeItem::m_PenColor			= QColor(160, 160, 160);
QColor	QNodeItem::m_PenColorHover		= QColor(50, 50, 50);
QColor	QNodeItem::m_PenColorSelected	= QColor(200, 50, 50);

QNodeItem::QNodeItem(QGraphicsItem* pParent, QNode* pNode, QTransferFunctionCanvas* pTransferFunctionCanvas) :
	QGraphicsEllipseItem(pParent),
	m_pTransferFunctionCanvas(pTransferFunctionCanvas),
	m_pNode(pNode),
	m_Cursor(),
	m_LastPos(),
	m_CachePen(),
	m_CacheBrush(),
	m_SuspendUpdate(false)
{
	// Styling
	setBrush(QBrush(QNodeItem::m_BackgroundColor));
	setPen(QPen(QNodeItem::m_PenColor, QNodeItem::m_PenWidth));

	// Make item movable
	setFlag(QGraphicsItem::ItemIsMovable);
	setFlag(QGraphicsItem::ItemSendsGeometryChanges);
	setFlag(QGraphicsItem::ItemIsSelectable);

	// We are going to catch hover events
	setAcceptHoverEvents(true);

	// Tooltip
	UpdateTooltip();
};

void QNodeItem::hoverEnterEvent(QGraphicsSceneHoverEvent* pEvent)
{
	QGraphicsEllipseItem::hoverEnterEvent(pEvent);

	// Don't overwrite styling when selected
	if (isSelected())
		return;

	// Change the cursor shape
	m_Cursor.setShape(Qt::CursorShape::PointingHandCursor);
	setCursor(m_Cursor);

	setPen(QPen(QNodeItem::m_PenColorHover, QNodeItem::m_PenWidthHover));
}

void QNodeItem::hoverLeaveEvent(QGraphicsSceneHoverEvent* pEvent)
{
	QGraphicsEllipseItem::hoverLeaveEvent(pEvent);

	// Don't overwrite styling when selected
	if (isSelected())
		return;

	// Change the cursor shape back to normal
	m_Cursor.setShape(Qt::CursorShape::ArrowCursor);
	setCursor(m_Cursor);

	setPen(QPen(QNodeItem::m_PenColor, QNodeItem::m_PenWidth));
}

void QNodeItem::mousePressEvent(QGraphicsSceneMouseEvent* pEvent)
{
	QGraphicsEllipseItem::mousePressEvent(pEvent);

	// Change the cursor shape to normal
	m_Cursor.setShape(Qt::CursorShape::ArrowCursor);
	setCursor(m_Cursor);
}

QVariant QNodeItem::itemChange(GraphicsItemChange Change, const QVariant& Value)
{
	QPointF NewScenePoint = Value.toPointF();
 
	if (!m_SuspendUpdate && Change == QGraphicsItem::ItemPositionChange)
	{
		QPointF NodeRangeMin = m_pTransferFunctionCanvas->TransferFunctionToScene(QPointF(m_pNode->GetMinX(), m_pNode->GetMinY()));
		QPointF NodeRangeMax = m_pTransferFunctionCanvas->TransferFunctionToScene(QPointF(m_pNode->GetMaxX(), m_pNode->GetMaxY()));

		NewScenePoint.setX(qMin(NodeRangeMax.x(), qMax(NewScenePoint.x(), NodeRangeMin.x())));
		NewScenePoint.setY(qMin(NodeRangeMin.y(), qMax(NewScenePoint.y(), NodeRangeMax.y())));

		return NewScenePoint;
	}

	if (!m_SuspendUpdate && Change == QGraphicsItem::ItemPositionHasChanged)
	{
		QPointF NewTfPoint = m_pTransferFunctionCanvas->SceneToTransferFunction(NewScenePoint);

		m_pTransferFunctionCanvas->m_AllowUpdateNodes = false;

		m_pNode->SetX(NewTfPoint.x());
		m_pNode->SetY(NewTfPoint.y());

		m_pTransferFunctionCanvas->m_AllowUpdateNodes = true;

		return NewScenePoint;
	}

    if (Change == QGraphicsItem::ItemSelectedHasChanged)
	{
		if (isSelected())
		{
			// Cache the old pen and brush
			m_CachePen		= pen();
			m_CacheBrush	= brush();

			setPen(QPen(QNodeItem::m_PenColorSelected, QNodeItem::m_PenWidthSelected));
		}
		else
		{
			// Restore pold pen and brush
			setPen(m_CachePen);
			setBrush(m_CacheBrush);
		}
	}

    return QGraphicsItem::itemChange(Change, Value);
}

void QNodeItem::paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget)
{
 	pPainter->setPen(pen());
 	pPainter->setBrush(brush());
 
 	pPainter->drawEllipse(rect());
 }

void QNodeItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* pEvent)
{
	QGraphicsEllipseItem::mouseReleaseEvent(pEvent);

	// Change the cursor shape to normal
	m_Cursor.setShape(Qt::CursorShape::ArrowCursor);
	setCursor(m_Cursor);
}

void QNodeItem::mouseMoveEvent(QGraphicsSceneMouseEvent* pEvent)
{
	QGraphicsItem::mouseMoveEvent(pEvent);
}

void QNodeItem::setPos(const QPointF& Pos)
{
	QGraphicsEllipseItem::setPos(Pos);

	QRectF EllipseRect;

	EllipseRect.setTopLeft(QPointF(-m_Radius, -m_Radius));
	EllipseRect.setWidth(2.0f * m_Radius);
	EllipseRect.setHeight(2.0f * m_Radius);

	setRect(EllipseRect);
}

void QNodeItem::UpdateTooltip(void)
{
	QString ToolTipString;

	const QString R = QString::number(m_pNode->GetColor().red());
	const QString G = QString::number(m_pNode->GetColor().green());
	const QString B = QString::number(m_pNode->GetColor().blue());

	ToolTipString.append("<table>");
		ToolTipString.append("<tr>");
			ToolTipString.append("<td>Node</td><td>:</td>");
			ToolTipString.append("<td>" + QString::number(1) + "</td>");
		ToolTipString.append("</tr>");
		ToolTipString.append("<tr>");
			ToolTipString.append("<td>Position</td><td> : </td>");
			ToolTipString.append("<td>" + QString::number(m_pNode->GetPosition()) + "</td>");
		ToolTipString.append("</tr>");
		ToolTipString.append("<tr>");
			ToolTipString.append("<td>Opacity</td><td> : </td>");
			ToolTipString.append("<td>" + QString::number(m_pNode->GetOpacity()) + "</td>");
		ToolTipString.append("</tr>");
		ToolTipString.append("<tr>");
			ToolTipString.append("<td>Color</td><td> : </td>");
			ToolTipString.append("<td style='color:rgb(" + R + ", " + G + ", " + B + ")'><b>");
				ToolTipString.append("<style type='text/css'>backgournd {color:red;}</style>");
				ToolTipString.append("[");
					ToolTipString.append(R + ", ");
					ToolTipString.append(G + ", ");
					ToolTipString.append(B);
				ToolTipString.append("]");
			ToolTipString.append("</td></b>");
		ToolTipString.append("</tr>");
	ToolTipString.append("</table>");

	// Update the tooltip
	setToolTip(ToolTipString);
}