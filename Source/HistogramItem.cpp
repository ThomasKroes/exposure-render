
// Precompiled headers
#include "Stable.h"

#include "HistogramItem.h"

QHistogramItem::QHistogramItem(QGraphicsItem* pParent) :
	QGraphicsRectItem(pParent),
	m_Histogram(),
	m_Brush(),
	m_Pen(),
	m_PolygonItem(this),
	m_Lines()
{
	setBrush(Qt::NoBrush);
	setPen(Qt::NoPen);
}

QHistogramItem::~QHistogramItem(void)
{
	for (int i = 0; i < m_Lines.size(); i++)
	{
		scene()->removeItem(m_Lines[i]);
		delete m_Lines[i];
	}

	m_Lines.clear();
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

	for (int i = 0; i < m_Lines.size(); i++)
	{
		scene()->removeItem(m_Lines[i]);
		delete m_Lines[i];
	}

	m_Lines.clear();

	QPolygonF Polygon;

	QPointF CachedCanvasPoint;

	// Set the gradient stops
	for (int i = 0; i < m_Histogram.GetBins().size(); i++)
	{
		// Compute polygon point in scene coordinates
		QPointF CanvasPoint;
		CanvasPoint.setX(rect().width() * ((float)i / (float)m_Histogram.GetBins().size()));
		
		if (m_Histogram.GetBins()[i] <= 0.0f)
			CanvasPoint.setY(rect().height());
		else
			CanvasPoint.setY(rect().height() - (rect().height() * logf((float)m_Histogram.GetBins()[i]) / (1.2f * logf((float)m_Histogram.GetMax()))));

		if (i == 0)
		{
			QPointF CenterCopy = CanvasPoint;

			CenterCopy.setY(rect().height());

			Polygon.append(CenterCopy);
		}

		Polygon.append(CanvasPoint);

		if (i == (m_Histogram.GetBins().size() - 1))
		{
			QPointF CenterCopy = CanvasPoint;

			CenterCopy.setY(rect().height());

			Polygon.append(CenterCopy);
		}

		QGraphicsLineItem* pLineItem = new QGraphicsLineItem(this);

		pLineItem->setLine(QLineF(CachedCanvasPoint, CanvasPoint));
		pLineItem->setPen(QPen(QColor::fromHsl(0, 30, 140)));

		m_Lines.append(pLineItem);

		CachedCanvasPoint = CanvasPoint;
	}

	QLinearGradient LinearGradient;

	LinearGradient.setStart(0, rect().bottom());
	LinearGradient.setFinalStop(0, rect().top());

	QGradientStops GradientStops;

	GradientStops.append(QGradientStop(0, QColor::fromHsl(0, 10, 150, 0)));
	GradientStops.append(QGradientStop(1, QColor::fromHsl(0, 100, 150, 150)));

	LinearGradient.setStops(GradientStops);

	// Update the polygon geometry
	m_PolygonItem.setPolygon(Polygon);
	m_PolygonItem.setBrush(QBrush(LinearGradient));
	m_PolygonItem.setPen(Qt::NoPen);
}