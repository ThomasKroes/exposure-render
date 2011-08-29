#pragma once

#include <QtGui>

class QTransferFunction;
class QNodeItem;
class QNode;
class QTransferFunctionCanvas;
class QTransferFunctionGradient;

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
		pPainter->setPen(QColor(50, 50, 50));
        pPainter->setFont(QFont("Arial", 7, 2));
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
	QTransferFunctionGradient*	m_pTransferFunctionGradient;
	float						m_MarginTop;
	float						m_MarginBottom;
	float						m_MarginLeft;
	float						m_MarginRight;
	QAxisLabel*					m_AxisLabelX;
	QAxisLabel*					m_AxisLabelY;
};