
// Precompiled headers
#include "Stable.h"

#include "NodeItem.h"
#include "TransferFunction.h"
#include "TransferFunctionCanvas.h"

QPen QEdgeItem::m_PenNormal		= QPen(QBrush(QColor::fromHsl(0, 100, 150)), 1.3);
QPen QEdgeItem::m_PenHighlight	= QPen(QBrush(QColor::fromHsl(0, 100, 150)), 1.3);
QPen QEdgeItem::m_PenDisabled	= QPen(QBrush(QColor::fromHsl(0, 0, 150)), 1.3);

QEdgeItem::QEdgeItem(QTransferFunctionCanvas* pTransferFunctionCanvas) :
	QGraphicsLineItem(pTransferFunctionCanvas),
	m_pTransferFunctionCanvas(pTransferFunctionCanvas)
{
	setPen(m_PenNormal);
};

void QEdgeItem::paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget)
{
	if (isEnabled())
	{
		if (isUnderMouse() || isSelected())
			setPen(m_PenHighlight);
		else
			setPen(m_PenNormal);
	}
	else
	{
		setPen(m_PenDisabled);
	}

	QGraphicsLineItem::paint(pPainter, pOption, pWidget);
}