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
	virtual void hoverEnterEvent(QGraphicsSceneHoverEvent* pEvent);
	virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent* pEvent);
	virtual void hoverMoveEvent(QGraphicsSceneHoverEvent* pEvent);

public:
	void Update(void);
	void UpdateBackground(void);
	void UpdateGrid(void);
	void UpdateHistogram(void);
	void UpdateEdges(void);
	void UpdateNodes(void);
	void UpdateGradient(void);
	void UpdatePolygon(void);
	void UpdateCrossHairs(void);

	QPointF SceneToTransferFunction(const QPointF& ScenePoint);
	QPointF TransferFunctionToScene(const QPointF& TfPoint);

protected:
	QGraphicsRectItem*				m_pBackgroundRectangle;
	QBrush							m_BackgroundBrush;
	QPen							m_BackgroundPen;
	QList<QGraphicsLineItem*>		m_GridLinesHorizontal;
	QPen							m_GridPenHorizontal;
	QPen							m_GridPenVertical;
	QGraphicsPolygonItem*			m_pPolygon;
	QLinearGradient					m_PolygonGradient;
	QGraphicsPolygonItem*			m_pHistogram;
	QGraphicsLineItem*				m_CrossHairH;
	QGraphicsLineItem*				m_CrossHairV;
	QGraphicsTextItem*				m_CrossHairText;
	bool							m_RealisticsGradient;
	bool							m_AllowUpdateNodes;

	// Nodes and edges
	QList<QNodeItem*>				m_NodeItems;
	QList<QGraphicsLineItem*>		m_EdgeItems;

	// Depth ordering
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