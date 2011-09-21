
// Precompiled headers
#include "Stable.h"

#include "BackgroundItem.h"

QBackgroundItem::QBackgroundItem(QGraphicsItem* pParent) :
	QGraphicsRectItem(pParent),
	m_BrushEnabled(QBrush(QColor::fromHsl(0, 0, 185))),
	m_BrushDisabled(QBrush(QColor::fromHsl(0, 0, 210))),
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
	pPainter->setRenderHint(QPainter::RenderHint::Antialiasing, false);

	if (isEnabled())
	{
		pPainter->setBrush(m_BrushEnabled);
		pPainter->setPen(m_PenEnabled);
	}
	else
	{
		pPainter->setBrush(m_BrushDisabled);
		pPainter->setPen(m_PenDisabled);
	}

	pPainter->drawRect(QRectF(0, 0, rect().width(), rect().height()));

	pPainter->setBrush(QBrush(Qt::red));
}