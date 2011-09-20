#pragma once

#include "TransferFunctionCanvas.h"
#include "TransferFunctionGradient.h"

class QTransferFunction;
class QNodeItem;
class QNode;

class QAxisLabel : public QGraphicsRectItem
{
public:
    QAxisLabel(QGraphicsItem* pParent, const QString& Text) :
		QGraphicsRectItem(pParent),
		m_Text(Text)
	{
	}

	virtual void QAxisLabel::paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget = NULL)
    {
		// Use anti aliasing
		pPainter->setRenderHints(QPainter::Antialiasing);

		pPainter->setPen(QColor(50, 50, 50));
        pPainter->setFont(QFont("Arial", 7));
        pPainter->drawText(rect(), Qt::AlignCenter, m_Text);
    }

	QString	m_Text;
};

class QBackgroundRectangle : public QGraphicsRectItem
{
public:
	QBackgroundRectangle(QGraphicsItem* pParent);

	QBackgroundRectangle::QBackgroundRectangle(const QBackgroundRectangle& Other)
	{
		*this = Other;
	};

	QBackgroundRectangle& operator = (const QBackgroundRectangle& Other)			
	{
		m_BrushEnabled	= Other.m_BrushEnabled;
		m_BrushDisabled	= Other.m_BrushDisabled;
		m_PenEnabled	= Other.m_PenEnabled;
		m_PenDisabled	= Other.m_PenDisabled;

		return *this;
	}

	virtual void paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget);

private:
	QBrush	m_BrushEnabled;
	QBrush	m_BrushDisabled;
	QPen	m_PenEnabled;
	QPen	m_PenDisabled;
};



class QHistogramItem : public QGraphicsRectItem
{
public:
	QHistogramItem(QGraphicsItem* pParent) :
		QGraphicsRectItem(pParent),
		m_PolygonItem(pParent),
		m_Brush(),
		m_Pen()
	{
	}

	QHistogramItem::QHistogramItem(const QHistogramItem& Other)
	{
		*this = Other;
	}

	QHistogramItem& operator = (const QHistogramItem& Other)			
	{
		m_Brush	= Other.m_Brush;
		m_Pen	= Other.m_Pen;

		return *this;
	}

	void SetHistogram(QHistogram& Histogram)
	{
		setVisible(Histogram.GetEnabled());

		if (!Histogram.GetEnabled())
			return;

		QPolygonF Polygon;

		// Set the gradient stops
		for (int i = 0; i < Histogram.GetBins().size(); i++)
		{
			// Compute polygon point in scene coordinates
			QPointF ScenePoint = QPointF(i, logf((float)gHistogram.GetBins()[i]) / logf(1.5f * (float)gHistogram.GetMax()));

			if (i == 0)
			{
				QPointF CenterCopy = ScenePoint;

				CenterCopy.setY(rect().height());

				Polygon.append(CenterCopy);
			}

			Polygon.append(ScenePoint);

			if (i == (gHistogram.GetBins().size() - 1))
			{
				QPointF CenterCopy = ScenePoint;

				CenterCopy.setY(rect().height());

				Polygon.append(CenterCopy);
			}
		}

		QLinearGradient LinearGradient;

		LinearGradient.setStart(0, rect().bottom());
		LinearGradient.setFinalStop(0, rect().top());

		QGradientStops GradientStops;

		GradientStops.append(QGradientStop(0, QColor::fromHsl(0, 100, 150, 0)));
		GradientStops.append(QGradientStop(1, QColor::fromHsl(0, 100, 150, 255)));

		LinearGradient.setStops(GradientStops);

		// Update the polygon geometry
		m_PolygonItem.setPolygon(Polygon);
		m_PolygonItem.setBrush(QBrush(LinearGradient));
	}

public:
	QGraphicsPolygonItem	m_PolygonItem;
	QBrush					m_Brush;
	QPen					m_Pen;
};






class QTFView : public QGraphicsView
{
	Q_OBJECT

public:
	QTFView(QWidget* pParent = NULL);

	void resizeEvent(QResizeEvent* pResizeEvent);

	void SetHistogram(QHistogram& Histogram);
	void UpdateHistogram(void);

private:
	QGraphicsScene			m_Scene;
	QBackgroundRectangle	m_Background;
	QHistogramItem			m_Histogram;
};

class QTransferFunctionView : public QGraphicsView
{
    Q_OBJECT

public:
    QTransferFunctionView(QWidget* pParent = NULL);

	void resizeEvent(QResizeEvent* pResizeEvent);
	void mousePressEvent(QMouseEvent* pEvent);

public slots:
	void OnNodeSelectionChanged(QNode* pNode);
	void OnHistogramChanged(void);

public:
	QGraphicsScene				m_GraphicsScene;
	QTransferFunctionCanvas		m_TransferFunctionCanvas;
	QAxisLabel					m_AxisLabelX;
	QAxisLabel					m_AxisLabelY;

	
};