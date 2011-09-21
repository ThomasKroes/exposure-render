#pragma once

class QGridItem : public QGraphicsRectItem
{
public:
	QGridItem(QGraphicsItem* pParent);
	QGridItem::QGridItem(const QGridItem& Other);
	QGridItem& operator = (const QGridItem& Other);

	virtual void paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget);

private:
	QBrush	m_BrushEnabled;
	QBrush	m_BrushDisabled;
	QPen	m_PenEnabled;
	QPen	m_PenDisabled;
	int		m_NumY;
	QFont	m_Font;
};