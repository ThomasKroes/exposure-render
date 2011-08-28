#pragma once

#include <QtGui>

class QTransferFunction;
class QNodeItem;
class QNode;

class QTransferFunctionCanvas : public QGraphicsRectItem, QObject
{
	Q_OBJECT

public:
    QTransferFunctionCanvas(QGraphicsItem* pParent);

protected:
	void	resizeEvent(QResizeEvent* pResizeEvent);
	void	mousePressEvent(QGraphicsSceneMouseEvent* pGraphicsSceneMouseEvent);

public slots:
	void	SetSelectedNode(QNode* pSelectedNode);
	void	Update(void);
	void	UpdateGrid(void);
	void	UpdateHistogram(void);
	void	UpdateEdges(void);
	void	UpdateNodes(void);
	void	UpdateNodeRanges(void);
	void	UpdateGradient(void);
	void	UpdatePolygon(void);

	QPointF SceneToTransferFunction(const QPointF& ScenePoint);
	QPointF TransferFunctionToScene(const QPointF& TfPoint);

private:
	QList<QNodeItem*>				m_Nodes;
	QList<QGraphicsLineItem*>		m_Edges;
	QGraphicsPolygonItem*			m_pPolygon;
	QLinearGradient					m_LinearGradient;
	QList<QGraphicsLineItem*>		m_GridLinesHorizontal;
	QPen							m_GridPenHorizontal;
};

class QTransferFunctionView : public QGraphicsView
{
    Q_OBJECT

public:
    QTransferFunctionView(QWidget* pParent);

	void	drawBackground(QPainter* pPainter, const QRectF& Rectangle);
	void	mousePressEvent(QMouseEvent* pEvent);

signals:
	void SelectionChange(QNode* pTransferFunctionNode);

private slots:
	void OnNodeAdd(QNode* pTransferFunctionNode);
	void OnNodeSelectionChanged(QNode* pNode);

public:
	QGraphicsScene*				m_pGraphicsScene;
	QTransferFunctionCanvas*	m_pTransferFunctionCanvas;
	float						m_Margin;
};