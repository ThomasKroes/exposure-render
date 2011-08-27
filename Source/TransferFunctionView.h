#pragma once

#include <QtGui>

class QTransferFunction;
class QNodeGraphics;

class QTransferFunctionView : public QGraphicsView
{
    Q_OBJECT

public:
    QTransferFunctionView(QWidget* pParent, QTransferFunction* pTransferFunction);

	void drawBackground(QPainter* pPainter, const QRectF& Rectangle);
	void resizeEvent(QResizeEvent* pResizeEvent);
//	void mouseMoveEvent(QMouseEvent * event);
	void keyPressEvent(QKeyEvent* pEvent);
	void keyReleaseEvent(QKeyEvent* pEvent);
	void mousePressEvent ( QMouseEvent * event );

	void OnNodeMove(QNodeGraphics* pNodeGraphics);

	void SetSelectedNode(QNode* pSelectedNode);

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
	void OnNodeRemove(QNode* pTransferFunctionNode);
	void OnNodeSelectionChanged(QNode* pNode);
	

public:
	QRectF							m_EditRect;

private:
	QGraphicsScene*					m_pGraphicsScene;
	QTransferFunction*				m_pTransferFunction;
	QList<QNodeGraphics*>			m_Nodes;
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