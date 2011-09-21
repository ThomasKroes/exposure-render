
// Precompiled headers
#include "Stable.h"

#include "GridItem.h"

QGridItem::QGridItem(QGraphicsItem* pParent) :
	QGraphicsRectItem(pParent),
	m_BrushEnabled(QBrush(QColor::fromHsl(0, 0, 170))),
	m_BrushDisabled(QBrush(QColor::fromHsl(0, 0, 210))),
	m_PenEnabled(QPen(QColor::fromHsl(0, 0, 140))),
	m_PenDisabled(QPen(QColor::fromHsl(0, 0, 160)))
{
}

QGridItem::QGridItem(const QGridItem& Other)
{
	*this = Other;
}

QGridItem& QGridItem::operator=(const QGridItem& Other)
{
	m_BrushEnabled	= Other.m_BrushEnabled;
	m_BrushDisabled	= Other.m_BrushDisabled;
	m_PenEnabled	= Other.m_PenEnabled;
	m_PenDisabled	= Other.m_PenDisabled;

	return *this;
}

void QGridItem::paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget)
{
	if (isEnabled())
	{
		setBrush(m_BrushEnabled);
		setPen(m_PenEnabled);
	}
	else
	{
		setBrush(m_BrushDisabled);
		setPen(m_PenDisabled);
	}

	QGraphicsRectItem::paint(pPainter, pOption, pWidget);
}