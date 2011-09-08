#pragma once

#include <QtGui>

class QNode;
class QTransferFunctionCanvas;

class QEdgeItem : public QGraphicsLineItem
{
public:
	QEdgeItem(QTransferFunctionCanvas* pTransferFunctionCanvas);

	QEdgeItem::QEdgeItem(const QEdgeItem& Other)
	{
		*this = Other;
	};

	QEdgeItem& operator = (const QEdgeItem& Other)			
	{
		m_pTransferFunctionCanvas = Other.m_pTransferFunctionCanvas;

		return *this;
	}

	void UpdateTooltip(void);
	
	void setPos(const QPointF& Pos);

public:
	QTransferFunctionCanvas*	m_pTransferFunctionCanvas;

	static QPen					m_PenNormal;
	static QPen					m_PenHighlight;
	static QPen					m_PenDisabled;
};