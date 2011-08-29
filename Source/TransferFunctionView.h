#pragma once

#include <QtGui>

class QTransferFunction;
class QNodeItem;
class QNode;
class QTransferFunctionCanvas;

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
		pPainter->setPen(QColor(75, 75, 75));
        pPainter->setFont(QFont("Arial", 7));
        pPainter->drawText(rect(), Qt::AlignCenter, m_Text);
    }

	QString	m_Text;
};

class QTransferFunctionView : public QGraphicsView
{
    Q_OBJECT

public:
    QTransferFunctionView(QWidget* pParent);

	void drawBackground(QPainter* pPainter, const QRectF& Rectangle);
	void resizeEvent(QResizeEvent* pResizeEvent);
	void mousePressEvent(QMouseEvent* pEvent);

private slots:
	void OnNodeSelectionChanged(QNode* pNode);
	void OnHistogramChanged(void);
	void Update(void);

public:
	QGraphicsScene*				m_pGraphicsScene;
	QTransferFunctionCanvas*	m_pTransferFunctionCanvas;
	float						m_Margin;
	QAxisLabel*					m_AxisLabelX;
	QAxisLabel*					m_AxisLabelY;
	QAxisLabel*					m_pMinX;
	QAxisLabel*					m_pMaxX;
};