#pragma once

class QTransferFunction;
class QNodeItem;
class QNode;

class QTransferFunctionGradient : public QGraphicsRectItem
{
public:
    QTransferFunctionGradient(QGraphicsItem* pParent, QGraphicsScene* pGraphicsScene);

public:
	void Update(void);

	QPointF SceneToTransferFunction(const QPointF& ScenePoint);
	QPointF TransferFunctionToScene(const QPointF& TfPoint);

protected:
	QList<QGraphicsRectItem*>	m_CheckerRectangles;
	QGraphicsRectItem*			m_pGradientRectangle;
	QLinearGradient				m_LinearGradient;

	friend class QTransferFunctionView;
	friend class QNodeItem;
};