
// Precompiled headers
#include "Stable.h"

#include "GridItem.h"

QGridItem::QGridItem(QGraphicsItem* pParent) :
	QGraphicsRectItem(pParent),
	m_BrushEnabled(QBrush(QColor::fromHsl(0, 0, 170))),
	m_BrushDisabled(QBrush(QColor::fromHsl(0, 0, 210))),
	m_PenEnabled(QPen(QColor::fromHsl(0, 0, 140), 0.1)),
	m_PenDisabled(QPen(QColor::fromHsl(0, 0, 190))),
	m_NumY(10)
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

	const float DY = rect().height() / (float)m_NumY;

	for (int i = 1; i < m_NumY; i++)
	{
		pPainter->drawLine(QPointF(0, i * DY), QPointF(rect().width(), i * DY));
	}
}