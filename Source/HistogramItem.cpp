
// Precompiled headers
#include "Stable.h"

#include "HistogramItem.h"

QHistogramItem::QHistogramItem(QGraphicsItem* pParent) :
	QGraphicsPolygonItem(pParent),
	m_Histogram(),
	m_Brush(),
	m_Pen(),
	m_Rectangle()
{
}

QHistogramItem::QHistogramItem(const QHistogramItem& Other)
{
	*this = Other;
}

QHistogramItem& QHistogramItem::operator=(const QHistogramItem& Other)
{
	m_Brush	= Other.m_Brush;
	m_Pen	= Other.m_Pen;

	return *this;
}

void QHistogramItem::SetHistogram(QHistogram& Histogram)
{
	m_Histogram = Histogram;
	Update();
}

void QHistogramItem::Update(void)
{
	setVisible(m_Histogram.GetEnabled());

	if (!m_Histogram.GetEnabled())
		return;

	QPolygonF Polygon;

	// Set the gradient stops
	for (int i = 0; i < m_Histogram.GetBins().size(); i++)
	{
		// Compute polygon point in scene coordinates
		QPointF ScenePoint;
		ScenePoint.setX(GetRectangle().width() * ((float)i / (float)m_Histogram.GetBins().size()));
		
		if (m_Histogram.GetBins()[i] <= 0.0f)
			ScenePoint.setY(GetRectangle().height());
		else
			ScenePoint.setY(GetRectangle().height() - (GetRectangle().height() * logf((float)m_Histogram.GetBins()[i]) / (1.2f * logf((float)m_Histogram.GetMax()))));

		if (i == 0)
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(GetRectangle().height());

			Polygon.append(CenterCopy);
		}

		Polygon.append(ScenePoint);

		if (i == (m_Histogram.GetBins().size() - 1))
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(GetRectangle().height());

			Polygon.append(CenterCopy);
		}
	}

	QLinearGradient LinearGradient;

	LinearGradient.setStart(0, GetRectangle().bottom());
	LinearGradient.setFinalStop(0, GetRectangle().top());

	QGradientStops GradientStops;

	GradientStops.append(QGradientStop(0, QColor::fromHsl(0, 60, 150, 0)));
	GradientStops.append(QGradientStop(1, QColor::fromHsl(0, 60, 150, 100)));

	LinearGradient.setStops(GradientStops);

	// Update the polygon geometry
	setPolygon(Polygon);
	setBrush(QBrush(LinearGradient));
	setPen(QPen(QColor::fromHsl(0, 60, 150)));
}

QRectF QHistogramItem::GetRectangle(void) const
{
	return m_Rectangle;
}

void QHistogramItem::SetRectangle(const QRectF& Rectangle)
{
	m_Rectangle = Rectangle;
	Update();
}