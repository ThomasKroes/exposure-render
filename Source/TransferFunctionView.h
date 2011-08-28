#pragma once

#include <QtGui>

class QTransferFunction;
class QNodeItem;
class QNode;

class QTransferFunctionCanvas : public QGraphicsRectItem
{
public:
    QTransferFunctionCanvas(QGraphicsItem* pParent, QGraphicsScene* pGraphicsScene);

protected:
	void	resizeEvent(QResizeEvent* pResizeEvent);

public:
	void	SetSelectedNode(QNode* pSelectedNode);
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
	

	friend class QTransferFunctionView;
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
};