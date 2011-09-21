#pragma once

#include "Histogram.h"

class QHistogramItem : public QGraphicsRectItem
{
public:
	QHistogramItem(QGraphicsItem* pParent);
	virtual ~QHistogramItem(void);
	QHistogramItem::QHistogramItem(const QHistogramItem& Other);
	QHistogramItem& operator = (const QHistogramItem& Other);

	void	SetHistogram(QHistogram& Histogram);
	void	Update(void);

public:
	QHistogram					m_Histogram;
	QBrush						m_Brush;
	QPen						m_Pen;
	QGraphicsPolygonItem		m_PolygonItem;
	QList<QGraphicsLineItem*>	m_Lines;
};