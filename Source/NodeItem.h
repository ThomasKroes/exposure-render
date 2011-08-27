#pragma once

#include <QtGui>

class QNode;
class QTransferFunctionView;

class QNodeItem : public QGraphicsEllipseItem
{
public:
	QNodeItem(QGraphicsItem* pParent, QNode* pTransferFunctionNode, QTransferFunctionView* pTransferFunctionView);

	void UpdateTooltip(void);
	
	QPointF GetCenter(void) const
	{
		return rect().center();
	}

	void SetCenter(const QPointF& Center)
	{
		setRect(QRectF(Center - 0.5f * QPointF(QNodeItem::m_Radius, QNodeItem::m_Radius), QSizeF(QNodeItem::m_Radius, QNodeItem::m_Radius)));
	}

protected:
	virtual void				hoverEnterEvent(QGraphicsSceneHoverEvent* pEvent);
	virtual void				hoverLeaveEvent(QGraphicsSceneHoverEvent* pEvent);
	virtual void				mousePressEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual QVariant			itemChange(GraphicsItemChange Change, const QVariant& Value);
	virtual void				mouseReleaseEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual void				mouseMoveEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual void				paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget = NULL);

public:
	QTransferFunctionView*		m_pTransferFunctionView;
	QNode*						m_pNode;
	QCursor						m_Cursor;
	QPointF						m_LastPos;
	QPen						m_CachePen;
	QBrush						m_CacheBrush;

	static float				m_Radius;
	static float				m_RadiusHover;
	static float				m_RadiusSelected;
	static QColor				m_BackgroundColor;
	static QColor				m_TextColor;
	static float				m_PenWidth;
	static float				m_PenWidthHover;
	static float				m_PenWidthSelected;
	static QColor				m_PenColor;
	static QColor				m_PenColorHover;
	static QColor				m_PenColorSelected;
};