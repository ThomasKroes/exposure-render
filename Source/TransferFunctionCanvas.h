#pragma once

#include "TransferFunction.h"
#include "NodeItem.h"
#include "EdgeItem.h"

class QGridLine : public QGraphicsLineItem
{
public:
	QGridLine(QTransferFunctionCanvas* pTransferFunctionCanvas);

	QGridLine::QGridLine(const QGridLine& Other)
	{
		*this = Other;
	};

	QGridLine& operator = (const QGridLine& Other)			
	{
		m_pTransferFunctionCanvas = Other.m_pTransferFunctionCanvas;

		return *this;
	}

private:
	QTransferFunctionCanvas*	m_pTransferFunctionCanvas;
};

class QTransferFunctionCanvas : public QGraphicsRectItem
{
public:
    QTransferFunctionCanvas(QGraphicsItem* pParent, QGraphicsScene* pGraphicsScene);
	virtual ~QTransferFunctionCanvas(void);

public:
	void Update(void);
	void UpdateBackground(void);
	void UpdateGrid(void);
	void UpdateHistogram(void);
	void UpdateEdges(void);
	void UpdateNodes(void);
	void UpdateGradient(void);
	void UpdatePolygon(void);

	QPointF SceneToTransferFunction(const QPointF& ScenePoint);
	QPointF TransferFunctionToScene(const QPointF& TfPoint);
	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);
protected:
	QGraphicsRectItem		m_BackgroundRectangle;
	QBrush					m_BackgroundBrush;
	QPen					m_BackgroundPen;
	QList<QGridLine*>		m_GridLines;
	QPen					m_GridPenHorizontal;
	QPen					m_GridPenVertical;
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