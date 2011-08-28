#pragma once

#include <QtGui>

class QTransferFunction;
class QNode;

class QGradientMarker : public QGraphicsLineItem
{
public:
    QGradientMarker(QGraphicsItem* pParent);

private:
	QGraphicsRectItem*			m_pGradientRectangle;
	QLinearGradient				m_LinearGradient;
	QGraphicsPolygonItem*		m_PolygonTop;
	QGraphicsPolygonItem*		m_PolygonBottom;
	QSize						m_PolygonSize;
	QBrush						m_Brush;
	QPen						m_Pen;
};

class QGradientView : public QGraphicsView
{
    Q_OBJECT

public:
    QGradientView(QWidget* pParent);

	void drawBackground(QPainter* pPainter, const QRectF& Rectangle);
	void resizeEvent(QResizeEvent* pResizeEvent);

	void UpdateGradientMarkers();

	QPointF SceneToTf(const QPointF& ScenePoint);
	QPointF TfToScene(const QPointF& TfPoint);

private slots:
	void Update(void);

private:
	QGraphicsScene*				m_pGraphicsScene;
	QSizeF						m_CheckerSize;
	QGraphicsRectItem*			m_pGradientRectangle;
	QLinearGradient				m_LinearGradient;
	QList<QGradientMarker*>		m_Markers;
};