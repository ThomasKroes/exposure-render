#pragma once

#include <QtGui>

class QTransferFunction;
class QNodeItem;
class QNode;

class QTransferFunctionView : public QGraphicsView
{
    Q_OBJECT

public:
    QTransferFunctionView(QWidget* pParent, QTransferFunction* pTransferFunction);

	void	drawBackground(QPainter* pPainter, const QRectF& Rectangle);
	void	resizeEvent(QResizeEvent* pResizeEvent);
	void	mousePressEvent ( QMouseEvent * event );
	void	SetSelectedNode(QNode* pSelectedNode);
	void	UpdateCanvas(void);
	void	UpdateGrid(void);
	void	UpdateHistogram(void);
	void	UpdateEdges(void);
	void	UpdateNodes(void);
	void	UpdateNodeRanges(void);
	void	UpdateGradient(void);
	void	UpdatePolygon(void);
	
	QPointF SceneToTf(const QPointF& ScenePoint);
	QPointF TfToScene(const QPointF& TfPoint);

signals:
	void SelectionChange(QNode* pTransferFunctionNode);

private slots:
	void Update(void);
	void OnNodeAdd(QNode* pTransferFunctionNode);
	void OnNodeSelectionChanged(QNode* pNode);
	

public:
	QRectF							m_EditRect;
	QGraphicsScene*					m_pGraphicsScene;
	QTransferFunction*				m_pTransferFunction;
	QList<QNodeItem*>				m_Nodes;
	QList<QGraphicsLineItem*>		m_Edges;
	QGraphicsPolygonItem*			m_pPolygon;
	QCursor							m_Cursor;
	QLinearGradient					m_LinearGradient;
	QGraphicsRectItem*				m_pCanvas;
	QGraphicsRectItem*				m_pOutline;
	float							m_Margin;
	QList<QGraphicsLineItem*>		m_GridLinesHorizontal;
	QPen							m_GridPenHorizontal;
	QGraphicsTextItem*				m_Text;
};