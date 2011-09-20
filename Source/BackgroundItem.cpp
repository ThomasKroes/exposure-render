
// Precompiled headers
#include "Stable.h"

#include "BackgroundItem.h"

QBackgroundItem::QBackgroundItem(QGraphicsItem* pParent) :
	QGraphicsRectItem(pParent),
	m_BrushEnabled(QBrush(QColor::fromHsl(0, 0, 170))),
	m_BrushDisabled(QBrush(QColor::fromHsl(0, 0, 200))),
	m_PenEnabled(QPen(QColor::fromHsl(0, 0, 140))),
	m_PenDisabled(QPen(QColor::fromHsl(0, 0, 160)))
{
}

QBackgroundItem::QBackgroundItem(const QBackgroundItem& Other)
{
	*this = Other;
}

QBackgroundItem& QBackgroundItem::operator=(const QBackgroundItem& Other)
{
	m_BrushEnabled	= Other.m_BrushEnabled;
	m_BrushDisabled	= Other.m_BrushDisabled;
	m_PenEnabled	= Other.m_PenEnabled;
	m_PenDisabled	= Other.m_PenDisabled;

	return *this;
}

void QBackgroundItem::paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget)
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