
// Precompiled headers
#include "Stable.h"

#include "NodeItem.h"
#include "TransferFunction.h"
#include "TransferFunctionItem.h"

#define NODE_POSITION_EPSILON 0.01f

float	QNodeItem::m_Radius			= 4.0f;
QBrush	QNodeItem::m_BrushNormal	= QBrush(QColor::fromHsl(0, 100, 150));
QBrush	QNodeItem::m_BrushHighlight	= QBrush(QColor::fromHsl(0, 130, 150));
QBrush	QNodeItem::m_BrushDisabled	= QBrush(QColor::fromHsl(0, 0, 230));

QPen	QNodeItem::m_PenNormal		= QPen(QBrush(QColor::fromHsl(0, 100, 100)), 1.0);
QPen	QNodeItem::m_PenHighlight	= QPen(QBrush(QColor::fromHsl(0, 150, 50)), 1.0);
QPen	QNodeItem::m_PenDisabled	= QPen(QBrush(QColor::fromHsl(0, 0, 200)), 1.0);

QBrush	QNodeItem::m_SelectionBrush	= QBrush(QColor::fromHsl(43, 150, 150, 255));
QPen	QNodeItem::m_SelectionPen	= QPen(QBrush(QColor::fromHsl(0, 150, 100, 150)), 1.0);

QNodeItem::QNodeItem(QTransferFunctionItem* pTransferFunctionItem, QNode* pNode) :
	QGraphicsEllipseItem(pTransferFunctionItem),
	m_pTransferFunctionItem(pTransferFunctionItem),
	m_pNode(pNode),
	m_Cursor(),
	m_LastPos(),
	m_CachePen(),
	m_CacheBrush(),
	m_SuspendUpdate(false)
{
	// Styling
	setBrush(QNodeItem::m_BrushNormal);
	setPen(QNodeItem::m_PenNormal);

	// Make item movable
	setFlag(QGraphicsItem::ItemIsMovable);
	setFlag(QGraphicsItem::ItemSendsGeometryChanges);
	setFlag(QGraphicsItem::ItemIsSelectable);

	// Tooltip
	UpdateTooltip();

	setParentItem(m_pTransferFunctionItem);
};

QNodeItem::QNodeItem(const QNodeItem& Other)
{
	*this = Other;
};

QNodeItem& QNodeItem::operator=(const QNodeItem& Other)
{
	m_pTransferFunctionItem		= Other.m_pTransferFunctionItem;
	m_pNode						= Other.m_pNode;
	m_Cursor					= Other.m_Cursor;
	m_LastPos					= Other.m_LastPos;
	m_CachePen					= Other.m_CachePen;
	m_CacheBrush				= Other.m_CacheBrush;
	m_pNodeID					= Other.m_pNodeID;
	m_SuspendUpdate				= Other.m_SuspendUpdate;

	return *this;
}

QVariant QNodeItem::itemChange(GraphicsItemChange Change, const QVariant& Value)
{
	QPointF NewScenePoint = Value.toPointF();
 
	if (!m_SuspendUpdate && Change == QGraphicsItem::ItemPositionChange)
	{
		const float Width	= m_pTransferFunctionItem->rect().width();
		const float Height	= m_pTransferFunctionItem->rect().height();

		QPointF NodeRangeMin = QPointF(m_pNode->GetMinX() * Width, m_pNode->GetMinY() * Height);
		QPointF NodeRangeMax = QPointF(m_pNode->GetMaxX() * Width, m_pNode->GetMaxY() * Height);

		NewScenePoint.setX(qMin(NodeRangeMax.x() - NODE_POSITION_EPSILON, qMax(NewScenePoint.x(), NodeRangeMin.x() + NODE_POSITION_EPSILON)));
		NewScenePoint.setY(qMin(NodeRangeMax.y(), qMax(NewScenePoint.y(), NodeRangeMin.y())));

		return NewScenePoint;
	}

	if (!m_SuspendUpdate && Change == QGraphicsItem::ItemPositionHasChanged)
	{
		m_pTransferFunctionItem->m_AllowUpdateNodes = false;

		m_pNode->SetIntensity(NewScenePoint.x() / m_pTransferFunctionItem->rect().width());
		m_pNode->SetOpacity(1.0f - (NewScenePoint.y() / m_pTransferFunctionItem->rect().height()));

		m_pTransferFunctionItem->m_AllowUpdateNodes = true;

		return NewScenePoint;
	}

    return QGraphicsItem::itemChange(Change, Value);
}

void QNodeItem::paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget)
{
	QBrush Brush;
	QPen Pen;

	if (isEnabled())
	{
		if (isUnderMouse() || isSelected())
		{
			Brush	= QNodeItem::m_BrushHighlight;
			Pen		= QNodeItem::m_PenHighlight;
		}
		else
		{
			Brush	= QNodeItem::m_BrushNormal;
			Pen		= QNodeItem::m_PenNormal;
		}

		if (isUnderMouse() || isSelected())
		{
			QRectF SelRect = rect();
			SelRect.adjust(-2.6, -2.6, 2.6, 2.6);

			pPainter->setBrush(QNodeItem::m_SelectionBrush);
			pPainter->setPen(QNodeItem::m_SelectionPen);
			pPainter->drawEllipse(SelRect);
		}
	}
	else
	{
		Brush	= QNodeItem::m_BrushDisabled;
		Pen		= QNodeItem::m_PenDisabled;
	}

	pPainter->setBrush(Brush);
	pPainter->setPen(Pen);
	pPainter->drawEllipse(rect());
}

void QNodeItem::mousePressEvent(QGraphicsSceneMouseEvent* pEvent)
{
	QGraphicsItem::mousePressEvent(pEvent);

	if (pEvent->button() == Qt::LeftButton)
	{
		if (gTransferFunction.GetNodes().indexOf(*m_pNode) == 0 || gTransferFunction.GetNodes().indexOf(*m_pNode) == gTransferFunction.GetNodes().size() - 1)
		{
			m_Cursor.setShape(Qt::SizeVerCursor);
			setCursor(m_Cursor);
		}
		else
		{
			m_Cursor.setShape(Qt::SizeAllCursor);
			setCursor(m_Cursor);
		}
	}
}

void QNodeItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* pEvent)
{
	QGraphicsEllipseItem::mouseReleaseEvent(pEvent);

	// Change the cursor shape to normal
	m_Cursor.setShape(Qt::ArrowCursor);
	setCursor(m_Cursor);
}

void QNodeItem::mouseMoveEvent(QGraphicsSceneMouseEvent* pEvent)
{
	QGraphicsItem::mouseMoveEvent(pEvent);
}

void QNodeItem::setPos(const QPointF& Pos)
{
	if (Pos == pos())
		return;

	QGraphicsEllipseItem::setPos(Pos);

	QRectF EllipseRect;

	EllipseRect.setTopLeft(QPointF(-m_Radius, -m_Radius));
	EllipseRect.setWidth(2.0f * m_Radius);
	EllipseRect.setHeight(2.0f * m_Radius);

	setRect(EllipseRect);
}

void QNodeItem::UpdateTooltip(void)
{
	/*
	QString ToolTipString;

	const QString R = QString::number(m_pNode->GetDiffuse().red());
	const QString G = QString::number(m_pNode->GetDiffuse().green());
	const QString B = QString::number(m_pNode->GetDiffuse().blue());

	ToolTipString.append("<table>");
		ToolTipString.append("<tr>");
			ToolTipString.append("<td>Node</td><td>:</td>");
			ToolTipString.append("<td>" + QString::number(1) + "</td>");
		ToolTipString.append("</tr>");
		ToolTipString.append("<tr>");
			ToolTipString.append("<td>Position</td><td> : </td>");
			ToolTipString.append("<td>" + QString::number(m_pNode->GetIntensity()) + "</td>");
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
	*/
}