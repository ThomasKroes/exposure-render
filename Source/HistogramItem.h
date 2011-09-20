#pragma once

#include "Histogram.h"

class QHistogramItem : public QGraphicsPolygonItem
{
public:
	QHistogramItem(QGraphicsItem* pParent);
	QHistogramItem::QHistogramItem(const QHistogramItem& Other);
	QHistogramItem& operator = (const QHistogramItem& Other);

	void	SetHistogram(QHistogram& Histogram);
	void	Update(void);
	QRectF	GetRectangle(void) const;
	void	SetRectangle(const QRectF& Rectangke);

public:
	QHistogram	m_Histogram;
	QBrush		m_Brush;
	QPen		m_Pen;
	QRectF		m_Rectangle;
};