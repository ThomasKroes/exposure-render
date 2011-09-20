#pragma once

class QBackgroundItem : public QGraphicsRectItem
{
public:
	QBackgroundItem(QGraphicsItem* pParent);
	QBackgroundItem::QBackgroundItem(const QBackgroundItem& Other);
	QBackgroundItem& operator = (const QBackgroundItem& Other);

	virtual void paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget);

private:
	QBrush	m_BrushEnabled;
	QBrush	m_BrushDisabled;
	QPen	m_PenEnabled;
	QPen	m_PenDisabled;
};