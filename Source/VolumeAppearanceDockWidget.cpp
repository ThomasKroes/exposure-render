
#include <QtGui>

#include "VolumeAppearanceDockWidget.h"
#include "TransferFunctionWidget.h"
#include "MainWindow.h"

#include <QtGui/QGraphicsView>

class CTfe;
class CEdge;

class CEdgeGradient : public QGraphicsPolygonItem
{
public:
	CEdgeGradient(CTfe* pCtfe, CEdge* pEdge) :
		m_pCtfe(pCtfe),
		m_pEdge(pEdge)
	{
		m_ID = CEdgeGradient::m_NoInstances++;

		// We are going to catch hover events
//		setAcceptHoverEvents(true);

		// Set tooltip
//		QString ToolTip;
//		ToolTip.sprintf("Edge %d\nDrag up/down to move edge", m_ID);
//		setToolTip(ToolTip);

//		setLine(m_pNode0->boundingRect().center().x(), m_pNode0->boundingRect().center().y(), m_pNode1->boundingRect().center().x(), m_pNode1->boundingRect().center().y());

		// Styling
//		setPen(QPen(CEdge::m_PenColor, CEdge::m_PenWidth, CEdge::m_PenStyle));

		setBrush(QBrush(Qt::blue));

		// Don't outline
		setPen(QPen(Qt::PenStyle::NoPen));

		Update();
	};

	QRectF boundingRect() const
	{
		return QGraphicsPolygonItem::boundingRect();
	}

	void Update(void)
	{
		/*
		// Create the polygon
		m_Polygon.append(QPointF(m_pEdge->m_pNode0->boundingRect().center().x(), 0.0f));
		m_Polygon.append(QPointF(m_pEdge->m_pNode0->boundingRect().center().x(), m_pEdge->m_pNode0->boundingRect().center().y()));
		m_Polygon.append(QPointF(m_pEdge->m_pNode1->boundingRect().center().x(), m_pEdge->m_pNode1->boundingRect().center().y()));
		m_Polygon.append(QPointF(m_pEdge->m_pNode1->boundingRect().center().x(), 0.0f));

		// Set it
		setPolygon(m_Polygon);

		// Create gradient
		QColor Col0 = m_pEdge->m_pNode0->m_Kd;
		QColor Col1 = m_pEdge->m_pNode1->m_Kd;
		
		// Make partially transparent
		Col0.setAlphaF(0.3f);
		Col1.setAlphaF(0.3f);

		// Start/end position of gradient
		m_Gradient.setStart(QPointF(m_pEdge->m_pNode0->boundingRect().center().x(), 0.0f));
		m_Gradient.setFinalStop(QPointF(m_pEdge->m_pNode1->boundingRect().center().x(), 0.0f));

		// Set color stops
		m_Gradient.setColorAt(0, Col0);
		m_Gradient.setColorAt(0.5, Col1);

		// Set the brush
		setBrush(QBrush(m_Gradient));
		*/

	}

	void paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* )
	{
		QGraphicsPolygonItem::paint(pPainter, pOption);
	}

	virtual void hoverEnterEvent(QGraphicsSceneHoverEvent* pEvent)
	{
		// Change the cursor shape
		m_Cursor.setShape(Qt::CursorShape::PointingHandCursor);
		setCursor(m_Cursor);

		setPen(QPen(CEdgeGradient::m_PenColorHover, CEdgeGradient::m_PenWidthHover, CEdgeGradient::m_PenStyleHover));
	}

	virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent* pEvent)
	{
		// Change the cursor shape back to normal
		m_Cursor.setShape(Qt::CursorShape::ArrowCursor);
		setCursor(m_Cursor);

		setPen(QPen(CEdgeGradient::m_PenColor, CEdgeGradient::m_PenWidth, CEdgeGradient::m_PenStyle));
	}

	virtual void mousePressEvent(QGraphicsSceneMouseEvent* pEvent)
	{
		// Change the cursor shape
		m_Cursor.setShape(Qt::CursorShape::SizeVerCursor);
		setCursor(m_Cursor);
	}

	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* pEvent)
	{
		// Change the cursor shape back to normal
		m_Cursor.setShape(Qt::CursorShape::ArrowCursor);
		setCursor(m_Cursor);
	}

protected:
	CTfe*					m_pCtfe;
	CEdge*				m_pEdge;
	QPolygonF				m_Polygon;
	int						m_ID;
	QCursor					m_Cursor;
	QLinearGradient			m_Gradient;

	static int				m_NoInstances;
	static float			m_PenWidth;
	static float			m_PenWidthHover;
	static QColor			m_PenColor;
	static QColor			m_PenColorHover;
	static Qt::PenStyle		m_PenStyle;
	static Qt::PenStyle		m_PenStyleHover;
};

int				CEdgeGradient::m_NoInstances		= 0;
float			CEdgeGradient::m_PenWidth		= 1.2f;
float			CEdgeGradient::m_PenWidthHover	= 1.2f;
QColor			CEdgeGradient::m_PenColor		= QColor(60, 60, 60);
QColor			CEdgeGradient::m_PenColorHover	= QColor(220, 220, 220);
Qt::PenStyle	CEdgeGradient::m_PenStyle		= Qt::PenStyle::SolidLine;
Qt::PenStyle	CEdgeGradient::m_PenStyleHover	= Qt::PenStyle::DashLine;







class CTfeGradient : public QGraphicsView
{
public:
	CTfeGradient(CTfe* pCtfe, QWidget *parent = NULL) :
		QGraphicsView(parent),
		m_pCtfe(pCtfe)
	{
		// Never show scrollbars
		setVerticalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);
		setHorizontalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);

		setSceneRect(0,0, 100, 20);

		// Create scene and apply
		m_pScene = new QGraphicsScene(this);
		setScene(m_pScene);

		// Turn antialiasing on
		setRenderHint(QPainter::Antialiasing);

		Update();
	};

	virtual void Update(void)
	{
	}

	void drawBackground(QPainter *painter, const QRectF &rect)
	{
		const int NumX = 10;//floorf(rect.width() / m_CheckerSize.width());

		painter->setPen(QPen(Qt::PenStyle::NoPen));
		painter->setBrush(QBrush(Qt::gray));

		QRectF R = rect;

		R.adjust(10, 10, -10, -10);

		painter->drawRect(R);//i * m_CheckerSize.width(), 0, m_CheckerSize.width(), m_CheckerSize.height());
		return;

		for (int i = 0; i < NumX; i++)
		{
			if (i % 2 == 0)
			{
				painter->drawRect(R);//i * m_CheckerSize.width(), 0, m_CheckerSize.width(), m_CheckerSize.height());
			}
			else
			{
				painter->drawRect(i * m_CheckerSize.width(), m_CheckerSize.height(), m_CheckerSize.width(), m_CheckerSize.height());
			}
		}

	}

	virtual void resizeEvent(QResizeEvent* pEvent)
	{
		Update();
	}

protected:
	CTfe*				m_pCtfe;
	QGraphicsScene*		m_pScene;
	QGraphicsRectItem*	m_pCheckerBoardRow0;
	QGraphicsRectItem*	m_pCheckerBoardRow1;

	static QSizeF		m_CheckerSize;
};


QSizeF CTfeGradient::m_CheckerSize = QSizeF(20.0f, 20.0f);


class CTfe : public QGraphicsView
{
public:
    CTfe(QWidget *parent = 0)
	{
		// Never show scrollbars
		setVerticalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);
		setHorizontalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);

		setSceneRect(0, 0, 100, 400);

		m_SizePolicy.setHorizontalPolicy(QSizePolicy::Ignored);
		m_SizePolicy.setVerticalPolicy(QSizePolicy::Preferred);

		setSizePolicy(m_SizePolicy);
		m_pScene = new QGraphicsScene(this);
		// m_pScene->setItemIndexMethod(QGraphicsScene::NoIndex);
		m_pScene->setSceneRect(0, 0, 400, 400);
		setScene(m_pScene);
//		setCacheMode(CacheBackground);
		setRenderHint(QPainter::Antialiasing);

		QGraphicsRectItem *rectItem = new QGraphicsRectItem(this->rect(), 0, m_pScene);
		rectItem->setPen(QPen(Qt::black, 0.0, Qt::SolidLine));
		rectItem->setBrush(QColor(130, 130, 130, 255));
		
		// Compute Y interval
		const float IntY = this->rect().height() / (float)m_NoLinesV;

		// Draw vertical grid lines
		for (int i = 0; i < m_NoLinesV; i++)
		{
			// Create grid line
			QGraphicsLineItem* pGridLineV = new QGraphicsLineItem(this->rect().left(), i * IntY, this->rect().right(), i * IntY);

			// Add the grid line
			m_pScene->addItem(pGridLineV);
		}
		
		/*
		CNode* pCtfNode0 = new CNode(this, 0.0f, 50.0f, Qt::red);
		CNode* pCtfNode1 = new CNode(this, 100.0f, 90.0f, Qt::green);
		CNode* pCtfNode2 = new CNode(this, 120.0f, 70.0f, Qt::yellow);
		CNode* pCtfNode3 = new CNode(this, 140.0f, 100.0f, Qt::blue);
		
		CEdge* pEdge1 = new CEdge(this, pCtfNode0, pCtfNode1);
		CEdge* pEdge2 = new CEdge(this, pCtfNode1, pCtfNode2);
		CEdge* pEdge3 = new CEdge(this, pCtfNode2, pCtfNode3);


		m_pScene->addItem(new CEdgeGradient(this, pEdge1));
		m_pScene->addItem(new CEdgeGradient(this, pEdge2));
		m_pScene->addItem(new CEdgeGradient(this, pEdge3));

		m_pScene->addItem(pEdge1);
		m_pScene->addItem(pEdge2);
		m_pScene->addItem(pEdge3);

		m_pScene->addItem(pCtfNode0);
		m_pScene->addItem(pCtfNode1);
		m_pScene->addItem(pCtfNode2);
		m_pScene->addItem(pCtfNode3);

		fitInView(rect());
		*/
	};

	void CTfe::drawBackground(QPainter *painter, const QRectF &rect)
	{
	}

protected:
	QGraphicsScene*		m_pScene;
	QSizePolicy			m_SizePolicy;

	static int			m_NoLinesV;
};

int CTfe::m_NoLinesV	= 10;






CVolumeAppearancePresetsWidget::CVolumeAppearancePresetsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_pGridLayout(NULL),
	m_pNameLabel(NULL),
	m_pPresetNameComboBox(NULL),
	m_pLoadPresetPushButton(NULL),
	m_pSavePresetPushButton(NULL),
	m_pRemovePresetPushButton(NULL),
	m_pRenamePresetPushButton(NULL),
	m_pLoadAction(NULL)
{
	setTitle("Presets");
	setToolTip("Transfer function presets");

	// Create grid layout
	m_pGridLayout = new QGridLayout();
	m_pGridLayout->setColumnStretch(0, 1);
	m_pGridLayout->setColumnStretch(1, 1);
	m_pGridLayout->setColumnStretch(2, 1);
	m_pGridLayout->setColumnStretch(3, 1);
	m_pGridLayout->setColumnStretch(4, 1);
	m_pGridLayout->setColumnStretch(5, 1);

	setLayout(m_pGridLayout);

	QSizePolicy SizePolicy;
	SizePolicy.setHorizontalPolicy(QSizePolicy::Fixed);
	SizePolicy.setVerticalPolicy(QSizePolicy::Maximum);
	SizePolicy.setHorizontalStretch(5);
	SizePolicy.setVerticalStretch(0);

	// Film width
	m_pNameLabel = new QLabel("Name");
	m_pGridLayout->addWidget(m_pNameLabel, 0, 0);

	m_pPresetNameComboBox = new QComboBox(this);
	m_pPresetNameComboBox->addItem("Medical");
	m_pPresetNameComboBox->addItem("Engineering");
	m_pPresetNameComboBox->setEditable(true);
	m_pGridLayout->addWidget(m_pPresetNameComboBox, 0, 1);

	m_pLoadPresetPushButton = new QPushButton("");
	m_pLoadPresetPushButton->setFixedWidth(20);
	m_pLoadPresetPushButton->setFixedHeight(20);
	m_pGridLayout->addWidget(m_pLoadPresetPushButton, 0, 2);

	m_pSavePresetPushButton = new QPushButton(">");
	m_pSavePresetPushButton->setFixedWidth(20);
	m_pSavePresetPushButton->setFixedHeight(20);
	m_pGridLayout->addWidget(m_pSavePresetPushButton, 0, 3);

	m_pRemovePresetPushButton = new QPushButton("-");
	m_pRemovePresetPushButton->setFixedWidth(20);
	m_pRemovePresetPushButton->setFixedHeight(20);
	m_pGridLayout->addWidget(m_pRemovePresetPushButton, 0, 4);

	m_pRenamePresetPushButton = new QPushButton(".");
	m_pRenamePresetPushButton->setFixedWidth(20);
	m_pRenamePresetPushButton->setFixedHeight(20);
	m_pGridLayout->addWidget(m_pRenamePresetPushButton, 0, 5);
}

void CVolumeAppearancePresetsWidget::CreateActions(void)
{
	m_pLoadAction = new QWidgetAction(this);
    m_pLoadAction->setStatusTip(tr("Load an existing transfer function"));
	m_pLoadAction->setToolTip(tr("Load an existing transfer function"));
	connect(m_pLoadAction, SIGNAL(triggered()), this, SLOT(Open()));
	m_pLoadPresetPushButton->addAction(m_pLoadAction);
	gpMainWindow->m_pFileMenu->addAction(m_pLoadAction);

}

CVolumeAppearanceWidget::CVolumeAppearanceWidget(QWidget* pParent) :
	QWidget(pParent),
	m_pMainLayout(NULL),
	m_pVolumeAppearancePresetsWidget(NULL)
{
	// Create vertical layout
	m_pMainLayout = new QVBoxLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
	setLayout(m_pMainLayout);

	// Volume appearance presets widget
	m_pVolumeAppearancePresetsWidget = new CVolumeAppearancePresetsWidget(this);
	m_pMainLayout->addWidget(m_pVolumeAppearancePresetsWidget);

	// Transfer function widget
	m_pTransferFunctionWidget = new CTransferFunctionWidget(this);
	m_pMainLayout->addWidget(m_pTransferFunctionWidget);
	
}

CVolumeAppearanceDockWidget::CVolumeAppearanceDockWidget(QWidget *parent) :
	QDockWidget(parent),
	m_pVolumeAppearanceWidget(NULL)
{
	setWindowTitle("Appearance");
	setToolTip("Volume Appearance");

	m_pVolumeAppearanceWidget = new CVolumeAppearanceWidget(this);

	setWidget(m_pVolumeAppearanceWidget);
}