
// Precompiled headers
#include "Stable.h"

#include "EdgeItem.h"
#include "TransferFunction.h"
#include "TransferFunctionItem.h"

QPen QEdgeItem::m_PenNormal		= QPen(QBrush(QColor::fromHsl(0, 100, 150)), 1.0);
QPen QEdgeItem::m_PenHighlight	= QPen(QBrush(QColor::fromHsl(0, 100, 150)), 1.0);
QPen QEdgeItem::m_PenDisabled	= QPen(QBrush(QColor::fromHsl(0, 0, 200)), 1.0);

QEdgeItem::QEdgeItem(QTransferFunctionItem* pTransferFunctionItem) :
	QGraphicsLineItem(pTransferFunctionItem),
	m_pTransferFunctionItem(pTransferFunctionItem)
{
	setPen(m_PenNormal);

	setParentItem(m_pTransferFunctionItem);
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