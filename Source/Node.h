#pragma once

#include <QtGui>

/*
class CNode : public QGraphicsEllipseItem
{
public:
	enum Constraint
	{
		None,
		Vertical,
		Horizontal
	};

    CNode(CTfe* pCtfe, float Position, float Opacity, QColor Kd) :
		m_pCtfe(pCtfe),
		m_Position(Position),
		m_Opacity(Opacity),
		m_Kd(Kd)
	{
		m_ID = CNode::m_NoInstances++;

		m_pEllipse = new QGraphicsEllipseItem(this);

		// Set tooltip
		QString ToolTip;
		ToolTip.sprintf("Node %d\nDrag up/down/left/right to move node", m_ID);
		setToolTip(ToolTip);

		setRect(QRectF(m_Position - CNode::m_Radius, m_Opacity - CNode::m_Radius, CNode::m_Radius, CNode::m_Radius));

		// We are going to catch hover events
		setAcceptHoverEvents(true);

		// Styling
		setBrush(QBrush(CNode::m_BackgroundColor));
		setPen(QPen(CNode::m_PenColor, CNode::m_PenWidth));

		setFlag(QGraphicsItem::ItemIsMovable);
	};

	void AddEdge(CEdge* pEdge)
	{
		m_Edges.append(pEdge);
	}

	void RemoveEdge(CEdge* pEdge)
	{
		m_Edges.remove(pEdge);
	}

	QVariant itemChange(GraphicsItemChange Change, const QVariant& Value)
	{
		if (Change == QGraphicsItem::ItemPositionChange)
		{
			foreach (CEdge* pEdge, m_Edges)
			{
				pEdge->UpdatePosition();
			}
		}

		return Value;
	}

	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *)
	{
		QGraphicsEllipseItem::paint(painter, option);
		
		QTextOption TextOption;

		TextOption.setAlignment(Qt::AlignCenter);

		QFont Font;

		Font.setFamily("Courier");
		Font.setPointSize(6);
		Font.setBold(true);

		painter->setBrush(QBrush(QColor(220, 220, 220)));
		painter->setFont(Font);
		painter->setPen(QPen(CNode::m_TextColor));

		QString ID;

		ID.sprintf("%d", m_ID + 1);

		painter->drawText(boundingRect(), ID, TextOption);
	}

	virtual void hoverEnterEvent(QGraphicsSceneHoverEvent* pEvent)
	{
		// Change the cursor shape
		m_Cursor.setShape(Qt::CursorShape::PointingHandCursor);
		setCursor(m_Cursor);

		setPen(QPen(CNode::m_PenColorHover, CNode::m_PenWidthHover));
	}

	virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent* pEvent)
	{
		// Change the cursor shape back to normal
		m_Cursor.setShape(Qt::CursorShape::ArrowCursor);
		setCursor(m_Cursor);

		setPen(QPen(CNode::m_PenColor, CNode::m_PenWidth));
	}

	virtual void mousePressEvent(QGraphicsSceneMouseEvent* pEvent)
	{
		// Change the cursor shape
		m_Cursor.setShape(Qt::CursorShape::sizeAllCursor);
		setCursor(m_Cursor);
	}

	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* pEvent)
	{
		// Change the cursor shape to normal
		m_Cursor.setShape(Qt::CursorShape::ArrowCursor);
		setCursor(m_Cursor);
	}

public:
	QColor					m_Kd;

protected:
	CTfe*					m_pCtfe;
	QGraphicsEllipseItem*	m_pEllipse;
	float					m_Position;
	float					m_Opacity;
	bool					m_Hovering;
	int						m_ID;
	QCursor					m_Cursor;
	bool					m_Deletable;
	QList<CEdge *>			m_Edges;

	static int				m_NoInstances;
	static float			m_Radius;
	static float			m_RadiusHover;
	static QColor			m_BackgroundColor;
	static QColor			m_TextColor;
	static float			m_PenWidth;
	static float			m_PenWidthHover;
	static QColor			m_PenColor;
	static QColor			m_PenColorHover;
};
*/