#pragma once

#include <QtGui>

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

class QTransferFunctionView : public QGraphicsView
{
    Q_OBJECT

public:
    QTransferFunctionView(QWidget* pParent = NULL);

	void drawBackground(QPainter* pPainter, const QRectF& Rectangle);
	void resizeEvent(QResizeEvent* pResizeEvent);
	void mousePressEvent(QMouseEvent* pEvent);

public slots:
	void OnNodeSelectionChanged(QNode* pNode);
	void OnHistogramChanged(void);
	void Update(void);
	void OnRenderBegin(void);
	void OnRenderEnd(void);

public:
	QGraphicsScene				m_GraphicsScene;
	QTransferFunctionCanvas		m_TransferFunctionCanvas;
	QTransferFunctionGradient	m_TransferFunctionGradient;
	float						m_MarginTop;
	float						m_MarginBottom;
	float						m_MarginLeft;
	float						m_MarginRight;
	QAxisLabel					m_AxisLabelX;
	QAxisLabel					m_AxisLabelY;
	QGraphicsColorizeEffect		m_GraphicsBlurEffect;
};