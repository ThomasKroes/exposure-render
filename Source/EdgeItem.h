#pragma once

class QTransferFunctionItem;

class QEdgeItem : public QGraphicsLineItem
{
public:
	QEdgeItem(QTransferFunctionItem* pTransferFunctionItem);

	QEdgeItem::QEdgeItem(const QEdgeItem& Other)
	{
		*this = Other;
	};

	QEdgeItem& operator = (const QEdgeItem& Other)			
	{
		m_pTransferFunctionItem = Other.m_pTransferFunctionItem;

		return *this;
	}

	void UpdateTooltip(void);
	
	void setPos(const QPointF& Pos);
	void paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget);

public:
	QTransferFunctionItem*	m_pTransferFunctionItem;
	static QPen				m_PenNormal;
	static QPen				m_PenHighlight;
	static QPen				m_PenDisabled;
};