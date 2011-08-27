#pragma once

#include <QtGui>

class QTransferFunction;
class QNode;

class QGradientView : public QGraphicsView
{
    Q_OBJECT

public:
    QGradientView(QWidget* pParent, QTransferFunction* pTransferFunction);

	void drawBackground(QPainter* pPainter, const QRectF& Rectangle);
	void resizeEvent(QResizeEvent* pResizeEvent);

private slots:
	void Update(void);
	void OnNodeAdd(QNode* pTransferFunctionNode);
	void OnNodeRemove(QNode* pTransferFunctionNode);

private:
	QGraphicsScene*			m_pGraphicsScene;
	QTransferFunction*		m_pTransferFunction;
	QSizeF					m_CheckerSize;
	QGraphicsRectItem*		m_pGradientRectangle;
	QLinearGradient			m_LinearGradient;
};