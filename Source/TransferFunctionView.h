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
	void	mousePressEvent(QGraphicsSceneMouseEvent* pGraphicsSceneMouseEvent);

public:
	void	SetSelectedNode(QNode* pSelectedNode);
	void	Update(void);
	void	UpdateBackground(void);
	void	UpdateGrid(void);
	void	UpdateHistogram(void);
	void	UpdateEdges(void);
	void	UpdateNodes(void);
	void	UpdateNodeRanges(void);
	void	UpdateGradient(void);
	void	UpdatePolygon(void);

	QPointF SceneToTransferFunction(const QPointF& ScenePoint);
	QPointF TransferFunctionToScene(const QPointF& TfPoint);

protected:
	QBrush							m_BackgroundBrush;
	QPen							m_BackgroundPen;
	QList<QNodeItem*>				m_Nodes;
	QList<QGraphicsLineItem*>		m_Edges;
	QGraphicsPolygonItem*			m_pPolygon;
	QLinearGradient					m_LinearGradient;
	QList<QGraphicsLineItem*>		m_GridLinesHorizontal;
	QPen							m_GridPenHorizontal;

	friend class QTransferFunctionView;
};

class QTransferFunctionView : public QGraphicsView
{
    Q_OBJECT

public:
    QTransferFunctionView(QWidget* pParent);

	void drawBackground(QPainter* pPainter, const QRectF& Rectangle);
	void resizeEvent(QResizeEvent* pResizeEvent);

private slots:
	void OnNodeAdd(QNode* pNode);
	void OnNodeSelectionChanged(QNode* pNode);

public:
	QGraphicsScene*				m_pGraphicsScene;
	QTransferFunctionCanvas*	m_pTransferFunctionCanvas;
	float						m_Margin;
};