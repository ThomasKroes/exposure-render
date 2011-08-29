#pragma once

#include <QtGui>

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
		// Experimental
//		pPainter->fillRect(rect(), QBrush(QColor(150, 150, 150, 150)));

		pPainter->setPen(QColor(75, 75, 75));
        pPainter->setFont(QFont("Arial", 7));
        pPainter->drawText(rect(), Qt::AlignCenter, m_Text);
    }

	QString	m_Text;
};

class QTransferFunctionCanvas : public QGraphicsRectItem
{
public:
    QTransferFunctionCanvas(QGraphicsItem* pParent, QGraphicsScene* pGraphicsScene);

protected:
	void	resizeEvent(QResizeEvent* pResizeEvent);

public:
//	void	SetHistogram();
	void	Update(void);
	void	UpdateBackground(void);
	void	UpdateGrid(void);
	void	UpdateHistogram(void);
	void	UpdateEdges(void);
	void	UpdateNodes(void);
	void	UpdateGradient(void);
	void	UpdatePolygon(void);

	QPointF SceneToTransferFunction(const QPointF& ScenePoint);
	QPointF TransferFunctionToScene(const QPointF& TfPoint);

protected:
	QBrush							m_BackgroundBrush;
	QPen							m_BackgroundPen;
	QList<QGraphicsLineItem*>		m_GridLinesHorizontal;
	QPen							m_GridPenHorizontal;
	QPen							m_GridPenVertical;
	QList<QNodeItem*>				m_NodeItems;
	QList<QGraphicsLineItem*>		m_EdgeItems;
	QGraphicsPolygonItem*			m_pPolygon;
	QLinearGradient					m_LinearGradient;
	bool							m_RealisticsGradient;
	bool							m_AllowUpdateNodes;

	int								m_BackgroundZ;
	int								m_GridZ;
	int								m_HistogramZ;
	int								m_EdgeZ;
	int								m_PolygonZ;
	int								m_NodeZ;
	int								m_CrossHairZ;

	friend class QTransferFunctionView;
	friend class QNodeItem;
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