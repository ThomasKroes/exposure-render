#pragma once

#include <QtGui>

#include "TransferFunction.h"

class CTransferFunctionView;
class QTransferFunction;
class QNode;

class CNodeGraphics : public QGraphicsEllipseItem
{
public:
	CNodeGraphics(QGraphicsItem* pParent, QNode* pTransferFunctionNode, CTransferFunctionView* pTransferFunctionView);

	QNode* GetNode(void)
	{
		return m_pNode;
	}
	
	void UpdateTooltip(void);
	
	QPointF GetCenter(void) const
	{
		return rect().center();
	}

	void SetCenter(const QPointF& Center)
	{
		setRect(QRectF(Center - 0.5f * QPointF(CNodeGraphics::m_Radius, CNodeGraphics::m_Radius), QSizeF(CNodeGraphics::m_Radius, CNodeGraphics::m_Radius)));
	}

protected:
	virtual void				hoverEnterEvent(QGraphicsSceneHoverEvent* pEvent);
	virtual void				hoverLeaveEvent(QGraphicsSceneHoverEvent* pEvent);
	virtual void				mousePressEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual QVariant			itemChange(GraphicsItemChange Change, const QVariant& Value);
	virtual void				mouseReleaseEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual void				mouseMoveEvent(QGraphicsSceneMouseEvent* pEvent);
	virtual void				paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget = NULL);

private:
	CTransferFunctionView*		m_pTransferFunctionView;
	QNode*						m_pNode;
	QCursor						m_Cursor;
	QPointF						m_LastPos;
	QPen						m_CachePen;
	QBrush						m_CacheBrush;

	static float				m_Radius;
	static float				m_RadiusHover;
	static float				m_RadiusSelected;
	static QColor				m_BackgroundColor;
	static QColor				m_TextColor;
	static float				m_PenWidth;
	static float				m_PenWidthHover;
	static float				m_PenWidthSelected;
	static QColor				m_PenColor;
	static QColor				m_PenColorHover;
	static QColor				m_PenColorSelected;
};

/*
class CTransferFunctionEdge : public QObject
{
	Q_OBJECT

public:
	CTransferFunctionEdge(CTransferFunctionView* pTransferFunctionView, QNode* pTransferFunctionNode1, QNode* pTransferFunctionNode2);

	QNode* GetTransferFunctionNode1(void)	{ return m_pTransferFunctionNode1; }
	QNode* GetTransferFunctionNode2(void)	{ return m_pTransferFunctionNode2; }

private slots:
	void Update(void);

private:
	CTransferFunctionView*	m_pTransferFunctionView;
	QNode*	m_pTransferFunctionNode1;
	QNode*	m_pTransferFunctionNode2;
	QCursor					m_Cursor;

	QGraphicsLineItem*		m_pLine;

	static float			m_Radius;
	static float			m_RadiusHover;
	static QColor			m_BackgroundColor;
	static QColor			m_TextColor;
	static float			m_PenWidth;
	static float			m_PenWidthHover;
	static QColor			m_PenColor;
	static QColor			m_PenColorHover;
};
*/

class CTransferFunctionView : public QGraphicsView
{
    Q_OBJECT

public:
    CTransferFunctionView(QWidget* pParent, QTransferFunction* pTransferFunction);

	void drawBackground(QPainter* pPainter, const QRectF& Rectangle);
	void resizeEvent(QResizeEvent* pResizeEvent);
//	void mouseMoveEvent(QMouseEvent * event);
	void keyPressEvent(QKeyEvent* pEvent);
	void keyReleaseEvent(QKeyEvent* pEvent);
	void mousePressEvent ( QMouseEvent * event );

	void OnNodeMove(CNodeGraphics* pNodeGraphics);

	void SetSelectedNode(QNode* pSelectedNode);

	void	UpdateCanvas(void);
	void	UpdateGrid(void);
	void	UpdateHistogram(void);
	void	UpdateEdges(void);
	void	UpdateNodes(void);
	void	UpdateGradient(void);
	void	UpdatePolygon(void);
	

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
	QList<CNodeGraphics*>			m_Nodes;
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

class CGradientView : public QGraphicsView
{
    Q_OBJECT

public:
    CGradientView(QWidget* pParent, QTransferFunction* pTransferFunction);

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

class CNodePropertiesWidget : public QWidget
{
    Q_OBJECT

public:
	CNodePropertiesWidget(QWidget* pParent, QTransferFunction* pTransferFunction);

private slots:
	void OnNodeSelectionChanged(QNode* pNode);
	void OnNodeSelectionChanged(const int& Index);
	void OnPreviousNode(void);
	void OnNextNode(void);
	void OnPositionChanged(const int& Position);
	void OnOpacityChanged(const int& Opacity);
	void OnColorChanged(const QColor& Color);
	void OnNodeAdd(QNode* pNode);
	void OnNodeRemove(QNode* pNode);
	void OnTransferFunctionChanged(void);

private:
	QTransferFunction*	m_pTransferFunction;
	QGridLayout*		m_pMainLayout;
	QLabel*				m_pSelectionLabel;
	QGridLayout*		m_pSelectionLayout;
	QComboBox*			m_pNodeSelectionComboBox;
	QPushButton*		m_pPreviousNodePushButton;
	QPushButton*		m_pNextNodePushButton;
	QLabel*				m_pPositionLabel;
	QSlider*			m_pPositionSlider;
	QSpinBox*			m_pPositionSpinBox;
	QLabel*				m_pOpacityLabel;
	QSlider*			m_pOpacitySlider;
	QSpinBox*			m_pOpacitySpinBox;
	QLabel*				m_pColorLabel;
	QComboBox*			m_pColorComboBox;
	QLabel*				m_pRoughnessLabel;
	QSlider*			m_pRoughnessSlider;
	QSpinBox*			m_pRoughnessSpinBox;
};

class CTransferFunctionWidget : public QGroupBox
{
    Q_OBJECT

public:
    CTransferFunctionWidget(QWidget* pParent = NULL);

protected:
	QGridLayout*				m_pMainLayout;
	QTransferFunction*			m_pTransferFunction;
	CTransferFunctionView*		m_pTransferFunctionView;
	CGradientView*				m_pGradientView;
	CNodePropertiesWidget*		m_pNodePropertiesWidget;
};