#pragma once

#include "TransferFunction.h"
#include "NodeItem.h"
#include "EdgeItem.h"

class QTransferFunctionCanvas : public QGraphicsRectItem
{
public:
    QTransferFunctionCanvas(QGraphicsItem* pParent, QGraphicsScene* pGraphicsScene);
	virtual ~QTransferFunctionCanvas(void);

public:
	void Update(void);
	void UpdateGrid(void);
	void UpdateEdges(void);
	void UpdateNodes(void);
	void UpdateGradient(void);
	void UpdatePolygon(void);

	QPointF SceneToTransferFunction(const QPointF& ScenePoint);
	QPointF TransferFunctionToScene(const QPointF& TfPoint);
	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);
protected:
	QGraphicsPolygonItem	m_Polygon;
	QLinearGradient			m_PolygonGradient;
	QGraphicsPolygonItem	m_Histogram;
	bool					m_RealisticsGradient;
	bool					m_AllowUpdateNodes;

	// Nodes and edges
	QList<QNodeItem*>		m_Nodes;
	QList<QEdgeItem*>		m_Edges;

	// Depth ordering
	int						m_BackgroundZ;
	int						m_GridZ;
	int						m_HistogramZ;
	int						m_EdgeZ;
	int						m_PolygonZ;
	int						m_NodeZ;
	int						m_CrossHairZ;

	QPixmap					m_PixMap;

	friend class QTransferFunctionView;
	friend class QNodeItem;
};