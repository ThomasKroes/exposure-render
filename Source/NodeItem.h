#pragma once

class QNode;
class QTransferFunctionItem;

class QNodeItem : public QGraphicsEllipseItem
{
public:
	QNodeItem(QTransferFunctionItem* pTransferFunctionItem, QNode* pNode);
	QNodeItem::QNodeItem(const QNodeItem& Other);
	QNodeItem& operator = (const QNodeItem& Other);

	void UpdateTooltip(void);
	
	void setPos(const QPointF& Pos);

protected:
	virtual QVariant itemChange(GraphicsItemChange Change, const QVariant& Value);
	virtual void mousePressEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual void mouseMoveEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual void paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget);

public:
	QTransferFunctionItem*	m_pTransferFunctionItem;
	QNode*					m_pNode;
	QCursor					m_Cursor;
	QPointF					m_LastPos;
	QPen					m_CachePen;
	QBrush					m_CacheBrush;
	QGraphicsTextItem*		m_pNodeID;
	bool					m_SuspendUpdate;

	static float			m_Radius;
	static QBrush			m_BrushNormal;
	static QBrush			m_BrushHighlight;
	static QBrush			m_BrushDisabled;
	static QPen				m_PenNormal;
	static QPen				m_PenHighlight;
	static QPen				m_PenDisabled;
};