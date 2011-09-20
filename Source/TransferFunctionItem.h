#pragma once

#include "TransferFunction.h"

class QNodeItem;
class QEdgeItem;

class QTransferFunctionItem : public QGraphicsRectItem
{
public:
	QTransferFunctionItem(QGraphicsItem* pParent);
	virtual ~QTransferFunctionItem(void);
	QTransferFunctionItem::QTransferFunctionItem(const QTransferFunctionItem& Other);
	QTransferFunctionItem& operator = (const QTransferFunctionItem& Other);

	virtual void paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget);

	void Update(void);

	void SetTransferFunction(QTransferFunction* pTransferFunction);

protected:
	QTransferFunction*		m_pTransferFunction;
	QBrush					m_BrushEnabled;
	QBrush					m_BrushDisabled;
	QGraphicsPolygonItem	m_PolygonItem;
	QPolygon				m_Polygon;
	QList<QNodeItem*>		m_Nodes;
	QList<QEdgeItem*>		m_Edges;
	bool					m_AllowUpdateNodes;

	friend class QNodeItem;
	friend class QTFView;
};