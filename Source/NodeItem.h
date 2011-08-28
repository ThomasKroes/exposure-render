#pragma once

#include <QtGui>

class QNode;
class QTransferFunctionCanvas;

class QNodeItem : public QGraphicsEllipseItem
{
public:
	QNodeItem(QGraphicsItem* pParent, QNode* pNode, QTransferFunctionCanvas* pTransferFunctionCanvas);

	void UpdateTooltip(void);
	
	void setPos(const QPointF& Pos);

protected:
	virtual void				hoverEnterEvent(QGraphicsSceneHoverEvent* pEvent);
	virtual void				hoverLeaveEvent(QGraphicsSceneHoverEvent* pEvent);
	virtual void				mousePressEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual QVariant			itemChange(GraphicsItemChange Change, const QVariant& Value);
	virtual void				mouseReleaseEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual void				mouseMoveEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual void				paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget);

public:
	QTransferFunctionCanvas*	m_pTransferFunctionCanvas;
	QNode*						m_pNode;
	QCursor						m_Cursor;
	QPointF						m_LastPos;
	QPen						m_CachePen;
	QBrush						m_CacheBrush;
	QGraphicsTextItem*			m_pNodeID;

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