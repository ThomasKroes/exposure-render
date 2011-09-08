#pragma once

#include <QtGui>

class QNode;
class QTransferFunctionCanvas;

class QNodeItem : public QGraphicsEllipseItem
{
public:
	QNodeItem(QTransferFunctionCanvas* pTransferFunctionCanvas, QNode* pNode);

	QNodeItem::QNodeItem(const QNodeItem& Other)
	{
		*this = Other;
	};

	QNodeItem& operator = (const QNodeItem& Other)			
	{
		m_pTransferFunctionCanvas	= Other.m_pTransferFunctionCanvas;
		m_pNode						= Other.m_pNode;
		m_Cursor					= Other.m_Cursor;
		m_LastPos					= Other.m_LastPos;
		m_CachePen					= Other.m_CachePen;
		m_CacheBrush				= Other.m_CacheBrush;
		m_pNodeID					= Other.m_pNodeID;
		m_SuspendUpdate				= Other.m_SuspendUpdate;

		return *this;
	}

	void UpdateTooltip(void);
	
	void setPos(const QPointF& Pos);

protected:
	virtual void hoverEnterEvent(QGraphicsSceneHoverEvent* pEvent);
	virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent* pEvent);
	virtual QVariant itemChange(GraphicsItemChange Change, const QVariant& Value);
	virtual void mousePressEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual void mouseMoveEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual void paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget);

public:
	QTransferFunctionCanvas*	m_pTransferFunctionCanvas;
	QNode*						m_pNode;
	QCursor						m_Cursor;
	QPointF						m_LastPos;
	QPen						m_CachePen;
	QBrush						m_CacheBrush;
	QGraphicsTextItem*			m_pNodeID;
	bool						m_SuspendUpdate;

	static float				m_Radius;
	static QBrush				m_BrushNormal;
	static QBrush				m_BrushHighlight;
	static QBrush				m_BrushDisabled;
	static QPen					m_PenNormal;
	static QPen					m_PenHighlight;
	static QPen					m_PenDisabled;
};