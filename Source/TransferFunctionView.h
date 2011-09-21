#pragma once

#include "TransferFunctionCanvas.h"
#include "TransferFunctionGradient.h"
#include "BackgroundItem.h"
#include "HistogramItem.h"
#include "TransferFunctionItem.h"

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

class QTFView : public QGraphicsView
{
	Q_OBJECT

public:
	QTFView(QWidget* pParent = NULL);

	void resizeEvent(QResizeEvent* pResizeEvent);
	void mousePressEvent(QMouseEvent* pEvent);

	void SetHistogram(QHistogram& Histogram);
	void setEnabled(bool Enabled);

public slots:
	void OnTransferFunctionChanged(void);
	void OnNodeSelectionChanged(QNode* pNode);

private:
	QMargin					m_Margin;
	QRect					m_CanvasRectangle;
	QGraphicsScene			m_Scene;
	QBackgroundItem			m_Background;
	QHistogramItem			m_HistogramItem;
	QTransferFunctionItem	m_TransferFunctionItem;
};