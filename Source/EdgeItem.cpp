
#include "NodeItem.h"
#include "TransferFunction.h"
#include "TransferFunctionCanvas.h"

QPen QEdgeItem::m_PenNormal		= QPen(QBrush(QColor::fromHsl(0, 130, 120)), 1.3);
QPen QEdgeItem::m_PenHighlight	= QPen(QBrush(QColor::fromHsl(0, 150, 130)), 1.3);
QPen QEdgeItem::m_PenDisabled	= QPen(QBrush(QColor::fromHsl(0, 0, 70)), 1.3);

QEdgeItem::QEdgeItem(QTransferFunctionCanvas* pTransferFunctionCanvas) :
	QGraphicsLineItem(pTransferFunctionCanvas),
	m_pTransferFunctionCanvas(pTransferFunctionCanvas)
{
	setPen(m_PenNormal);
};