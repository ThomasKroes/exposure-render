#pragma once

#include "TransferFunctionCanvas.h"
#include "TransferFunctionGradient.h"
#include "HistogramItem.h"

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

class QTFView : public QGraphicsView
{
	Q_OBJECT

public:
	QTFView(QWidget* pParent = NULL);

	void resizeEvent(QResizeEvent* pResizeEvent);

	void SetHistogram(QHistogram& Histogram);

private:
	QMargin					m_Margin;
	QGraphicsScene			m_Scene;
	QBackgroundRectangle	m_Background;
	QHistogramItem			m_HistogramItem;
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