#pragma once

#include <QtGui>

class CEdge : public QGraphicsLineItem
{
public:
    CEdge(CTfe* pCtfe, CNode* pNode0, CNode* pNode1) :
		m_pCtfe(pCtfe),
		m_pNode0(pNode0),
		m_pNode1(pNode1)
	{
		m_ID = CEdge::m_NoInstances++;

		// We are going to catch hover events
		setAcceptHoverEvents(true);

		// Set tooltip
		QString ToolTip;
		ToolTip.sprintf("Edge %d\nDrag up/down to move edge", m_ID);
		setToolTip(ToolTip);

		UpdatePosition();

		// Styling
		setPen(QPen(CEdge::m_PenColor, CEdge::m_PenWidth, CEdge::m_PenStyle));
	};

	QRectF boundingRect() const
	{
		return QGraphicsLineItem::boundingRect();
	}

	void UpdatePosition(void)
	{
		setLine(m_pNode0->boundingRect().center().x(), m_pNode0->boundingRect().center().y(), m_pNode1->boundingRect().center().x(), m_pNode1->boundingRect().center().y());
	}

	void paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* )
	{
		QGraphicsLineItem::paint(pPainter, pOption);
	}

	virtual void hoverEnterEvent(QGraphicsSceneHoverEvent* pEvent)
	{
		// Change the cursor shape
		m_Cursor.setShape(Qt::CursorShape::PointingHandCursor);
		setCursor(m_Cursor);

		setPen(QPen(CEdge::m_PenColorHover, CEdge::m_PenWidthHover, CEdge::m_PenStyleHover));
	}

	virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent* pEvent)
	{
		// Change the cursor shape back to normal
		m_Cursor.setShape(Qt::CursorShape::ArrowCursor);
		setCursor(m_Cursor);

		setPen(QPen(CEdge::m_PenColor, CEdge::m_PenWidth, CEdge::m_PenStyle));
	}

	virtual void mousePressEvent(QGraphicsSceneMouseEvent* pEvent)
	{
		// Change the cursor shape
		m_Cursor.setShape(Qt::CursorShape::SizeVerCursor);
		setCursor(m_Cursor);
	}

	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* pEvent)
	{
		// Change the cursor shape back to normal
		m_Cursor.setShape(Qt::CursorShape::ArrowCursor);
		setCursor(m_Cursor);
	}

public:
	CNode*				m_pNode0;
	CNode*				m_pNode1;

protected:
	CTfe*					m_pCtfe;
	QGraphicsLineItem*		m_pLine;
	int						m_ID;
	QCursor					m_Cursor;

	static int				m_NoInstances;
	static float			m_PenWidth;
	static float			m_PenWidthHover;
	static QColor			m_PenColor;
	static QColor			m_PenColorHover;
	static Qt::PenStyle		m_PenStyle;
	static Qt::PenStyle		m_PenStyleHover;
};