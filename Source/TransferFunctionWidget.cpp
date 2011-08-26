
#include "TransferFunctionWidget.h"

float	CNodeGraphics::m_Radius				= 10.0f;
float	CNodeGraphics::m_RadiusHover		= 10.0f;
float	CNodeGraphics::m_RadiusSelected		= 10.0f;
QColor	CNodeGraphics::m_BackgroundColor	= QColor(230, 230, 230);
QColor	CNodeGraphics::m_TextColor			= QColor(20, 20, 20);
float	CNodeGraphics::m_PenWidth			= 1.7f;
float	CNodeGraphics::m_PenWidthHover		= 1.7f;
float	CNodeGraphics::m_PenWidthSelected	= 1.7f;
QColor	CNodeGraphics::m_PenColor			= QColor(160, 160, 160);
QColor	CNodeGraphics::m_PenColorHover		= QColor(50, 50, 50);
QColor	CNodeGraphics::m_PenColorSelected	= QColor(200, 50, 50);

/*
bool NodeLessThan(const QNode* pNodeA, const QNode* pNodeB)
{
   return pNodeA->GetPosition() < pNodeB->GetPosition();
}
*/

CNodeGraphics::CNodeGraphics(QGraphicsItem* pParent, QNode* pNode, CTransferFunctionView* pTransferFunctionView) :
	QGraphicsEllipseItem(pParent),
	m_pTransferFunctionView(pTransferFunctionView),
	m_pNode(pNode),
	m_Cursor(),
	m_LastPos(),
	m_CachePen(),
	m_CacheBrush()
{
	// Styling
	setBrush(QBrush(CNodeGraphics::m_BackgroundColor));
	setPen(QPen(CNodeGraphics::m_PenColor, CNodeGraphics::m_PenWidth));

	// Make item movable
	setFlag(QGraphicsItem::ItemIsMovable);
//	setFlag(QGraphicsItem::ItemSendsGeometryChanges);
	setFlag(QGraphicsItem::ItemIsSelectable);

	// We are going to catch hover events
	setAcceptHoverEvents(true);

	// Tooltip
	UpdateTooltip();
};

void CNodeGraphics::hoverEnterEvent(QGraphicsSceneHoverEvent* pEvent)
{
	QGraphicsEllipseItem::hoverEnterEvent(pEvent);

	// Don't overwrite styling when selected
	if (isSelected())
		return;

	// Change the cursor shape
	m_Cursor.setShape(Qt::CursorShape::PointingHandCursor);
	setCursor(m_Cursor);

	setPen(QPen(CNodeGraphics::m_PenColorHover, CNodeGraphics::m_PenWidthHover));
}

void CNodeGraphics::hoverLeaveEvent(QGraphicsSceneHoverEvent* pEvent)
{
	QGraphicsEllipseItem::hoverLeaveEvent(pEvent);

	// Don't overwrite styling when selected
	if (isSelected())
		return;

	// Change the cursor shape back to normal
	m_Cursor.setShape(Qt::CursorShape::ArrowCursor);
	setCursor(m_Cursor);

	setPen(QPen(CNodeGraphics::m_PenColor, CNodeGraphics::m_PenWidth));
}

void CNodeGraphics::mousePressEvent(QGraphicsSceneMouseEvent* pEvent)
{
//	m_LastPos = pos();
//    update();

	QGraphicsEllipseItem::mousePressEvent(pEvent);

	/*
	// Change the cursor shape
	if (m_pNode->GetAllowMoveH() && m_pNode->GetAllowMoveV())
		m_Cursor.setShape(Qt::CursorShape::sizeAllCursor);

	if (!m_pNode->GetAllowMoveH() && m_pNode->GetAllowMoveV())
		m_Cursor.setShape(Qt::CursorShape::SizeVerCursor);

	if (m_pNode->GetAllowMoveH() && !m_pNode->GetAllowMoveV())
		m_Cursor.setShape(Qt::CursorShape::SizeHorCursor);

	setCursor(m_Cursor);
	
	m_LastPos = pEvent->pos();
	

	m_pTransferFunctionView->SetSelectedNode(m_pNode);*/
}


QVariant CNodeGraphics::itemChange(GraphicsItemChange Change, const QVariant& Value)
{
    QPointF NewPoint = Value.toPointF();
 
    if (Change == ItemPositionChange && scene())
	{/*
		const float NormX = (GetCenter().x() - m_pTransferFunctionView->m_EditRect.left()) / m_pTransferFunctionView->m_EditRect.width();
		const float NormY = (GetCenter().y() - m_pTransferFunctionView->m_EditRect.top()) / m_pTransferFunctionView->m_EditRect.height();

		m_pNode->SetNormalizedX(NormX);
		m_pNode->SetNormalizedY(NormY);
		
//		UpdateTooltip();

//		m_pNode->SetValueY(0.01f * NewPoint.y());

		
		if (!m_pNode->GetAllowMoveH() && !m_pNode->GetAllowMoveV())
		{
			NewPoint = m_LastPos;
		}
		
		if (m_pNode->GetAllowMoveH() && !m_pNode->GetAllowMoveV())
			NewPoint.setY(m_LastPos.y());

		if (!m_pNode->GetAllowMoveH() && m_pNode->GetAllowMoveV())
			NewPoint.setX(m_LastPos.x());

		
		QRectF SceneRectangle = scene()->sceneRect();
		
		if (!SceneRectangle.contains(NewPoint))
		{
			// Keep the item inside the scene rect.
			NewPoint.setX(qMin(SceneRectangle.right(), qMax(NewPoint.x(), SceneRectangle.left())));
			NewPoint.setY(qMin(SceneRectangle.bottom(), qMax(NewPoint.y(), SceneRectangle.top())));
		}

		return NewPoint;*/
    }

	if (Change == QGraphicsItem::ItemSelectedHasChanged)
	{
		if (isSelected())
		{
			// Cache the old pen and brush
			m_CachePen		= pen();
			m_CacheBrush	= brush();

			setPen(QPen(CNodeGraphics::m_PenColorSelected, CNodeGraphics::m_PenWidthSelected));
		}
		else
		{
			// Restore pold pen and brush
			setPen(m_CachePen);
			setBrush(m_CacheBrush);
		}
	}
	

    return QGraphicsItem::itemChange(Change, Value);
}


void CNodeGraphics::mouseReleaseEvent(QGraphicsSceneMouseEvent* pEvent)
{
	QGraphicsEllipseItem::mouseReleaseEvent(pEvent);

	// Change the cursor shape to normal
//	m_Cursor.setShape(Qt::CursorShape::ArrowCursor);
//	setCursor(m_Cursor);
}

void CNodeGraphics::mouseMoveEvent(QGraphicsSceneMouseEvent* pEvent)
{
//	QGraphicsItem::mouseMoveEvent(pEvent);
	m_pNode->SetNormalizedX((pEvent->scenePos().x() - (float)m_pTransferFunctionView->rect().left()) / (float)m_pTransferFunctionView->rect().width());
	m_pNode->SetNormalizedY(1.0f - ((pEvent->scenePos().y() - (float)m_pTransferFunctionView->rect().top()) / (float)m_pTransferFunctionView->rect().height()));
}

void CNodeGraphics::paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget)
{
	pPainter->setPen(pen());
	pPainter->setBrush(brush());

	pPainter->drawEllipse(rect());
}

void CNodeGraphics::UpdateTooltip(void)
{
	QString ToolTipString;

	const QString R = QString::number(m_pNode->GetColor().red());
	const QString G = QString::number(m_pNode->GetColor().green());
	const QString B = QString::number(m_pNode->GetColor().blue());

	ToolTipString.append("<table>");
		ToolTipString.append("<tr>");
			ToolTipString.append("<td>Node</td><td>:</td>");
			ToolTipString.append("<td>" + QString::number(1) + "</td>");
		ToolTipString.append("</tr>");
		ToolTipString.append("<tr>");
			ToolTipString.append("<td>Position</td><td> : </td>");
			ToolTipString.append("<td>" + QString::number(m_pNode->GetPosition()) + "</td>");
		ToolTipString.append("</tr>");
		ToolTipString.append("<tr>");
			ToolTipString.append("<td>Opacity</td><td> : </td>");
			ToolTipString.append("<td>" + QString::number(m_pNode->GetOpacity()) + "</td>");
		ToolTipString.append("</tr>");
		ToolTipString.append("<tr>");
			ToolTipString.append("<td>Color</td><td> : </td>");
			ToolTipString.append("<td style='color:rgb(" + R + ", " + G + ", " + B + ")'><b>");
				ToolTipString.append("<style type='text/css'>backgournd {color:red;}</style>");
				ToolTipString.append("[");
					ToolTipString.append(R + ", ");
					ToolTipString.append(G + ", ");
					ToolTipString.append(B);
				ToolTipString.append("]");
			ToolTipString.append("</td></b>");
		ToolTipString.append("</tr>");
	ToolTipString.append("</table>");

	// Update the tooltip
	setToolTip(ToolTipString);
}
/*
CTransferFunctionEdge::CTransferFunctionEdge(CTransferFunctionView* pTransferFunctionView, QNode* pTransferFunctionNode1, QNode* pTransferFunctionNode2) :
	QObject(NULL),
	m_pTransferFunctionView(pTransferFunctionView),
	m_pTransferFunctionNode1(pTransferFunctionNode1),
	m_pTransferFunctionNode2(pTransferFunctionNode2),
	m_Cursor()
{
	m_pLine = new QGraphicsLineItem(NULL, m_pTransferFunctionView->scene());
	
	// Setup connections
	connect(pTransferFunctionNode1, SIGNAL(NodeMove()), this, SLOT(Update()));
};

void CTransferFunctionEdge::Update(void)
{
	m_pLine->setLine(m_pTransferFunctionNode1->GetPosition(), m_pTransferFunctionNode1->GetOpacity(), m_pTransferFunctionNode2->GetPosition(), m_pTransferFunctionNode2->GetOpacity());
}
*/

CTransferFunctionView::CTransferFunctionView(QWidget* pParent, QTransferFunction* pTransferFunction) :
	QGraphicsView(pParent),
	m_pGraphicsScene(NULL),
	m_pTransferFunction(pTransferFunction),
	m_pPolygon(NULL),
	m_pOutline(NULL),
	m_pCanvas(NULL),
	m_Margin(0.0f)
{
	// Dimensions
	setFixedHeight(170);

	// Styling
	setFrameShadow(QFrame::Shadow::Sunken);
	setFrameShape(NoFrame);

	// Never show scrollbars
	setVerticalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);
	setHorizontalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);

	// Status and tooltip
	setStatusTip("Transfer function editor");
	setToolTip("Transfer function editor");

	// Create scene and apply
	m_pGraphicsScene = new QGraphicsScene(this);
	setScene(m_pGraphicsScene);

	// Turn antialiasing on
	setRenderHint(QPainter::Antialiasing);

	// Setup connections
	connect(m_pTransferFunction, SIGNAL(FunctionChanged()), this, SLOT(Update()));
	connect(m_pTransferFunction, SIGNAL(NodeAdd(QNode*)), this, SLOT(OnNodeAdd(QNode*)));

	// Respond to changes in node selection
	connect(m_pTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));

	// Add polygon graphics item
	m_pPolygon = new QGraphicsPolygonItem;

	// Ensure the item is drawn in the right order
	m_pPolygon->setZValue(500);

	// Add the polygon
	m_pGraphicsScene->addItem(m_pPolygon);

	// Create rectangle for outline
	m_pOutline = new QGraphicsRectItem;
	m_pOutline->setRect(rect());
	m_pOutline->setBrush(QBrush(Qt::BrushStyle::NoBrush));
	m_pOutline->setPen(QPen(Qt::darkGray));

	// Ensure the item is drawn in the right order
	m_pOutline->setZValue(1000);

	// Add the rectangle
//	m_pGraphicsScene->addItem(m_pOutline);

	// Create canvas
	m_pCanvas = new QGraphicsRectItem;
	m_pCanvas->setRect(rect());
	m_pCanvas->setBrush(QBrush(QColor(190, 190, 190)));
	m_pCanvas->setPen(QPen(Qt::darkGray));

	// Add the rectangle
	m_pGraphicsScene->addItem(m_pCanvas);

	// Render
	Update();

	// Configure pen
	m_GridPenHorizontal.setStyle(Qt::PenStyle::DashLine);
	m_GridPenHorizontal.setColor(QColor(75, 75, 75, 120));
//	m_GridPenHorizontal.setWidthF(1.0f);

	m_Text = new QGraphicsTextItem();
	m_Text->setPos(m_EditRect.bottomLeft());
	m_Text->setTextWidth(m_EditRect.width());
	m_Text->setHtml("<center>Density</center>");

	m_pGraphicsScene->addItem(m_Text);

}

void CTransferFunctionView::drawBackground(QPainter* pPainter, const QRectF& Rectangle)
{
	
	QGraphicsView::drawBackground(pPainter, Rectangle);
	
	setBackgroundBrush(QBrush(QColor(240, 240, 240)));
/*
	setSceneRect(Rectangle);
	centerOn(Rectangle.center());

	const int NumX = m_pTransferFunction->m_Range / 50;

	QPen Pen;
	Pen.setStyle(Qt::PenStyle::SolidLine);
	Pen.setWidthF(0.1f);

	pPainter->setPen(Pen);
	pPainter->setBrush(QBrush(Qt::gray));

	for (int i = 0; i < NumX; i++)
	{
		pPainter->drawLine(QPointF(i * 50.0, GetRect().bottom()), QPointF(i * 50.0, GetRect().top()));
	}
	*/

	
}

void CTransferFunctionView::resizeEvent(QResizeEvent* pResizeEvent)
{
	QGraphicsView::resizeEvent(pResizeEvent);

	Update();
//	UpdateGrid();
}

void CTransferFunctionView::keyPressEvent(QKeyEvent* pEvent)
{
//	m_Cursor.setShape(Qt::CrossCursor);

//	setCursor(m_Cursor);
}

void CTransferFunctionView::keyReleaseEvent(QKeyEvent* pEvent)
{
}

void CTransferFunctionView::mousePressEvent(QMouseEvent* pEvent)
{
	QGraphicsView::mousePressEvent(pEvent);

	CNodeGraphics* pNodeGraphics = dynamic_cast<CNodeGraphics*>(itemAt(pEvent->pos()));

	if (!pNodeGraphics)
		SetSelectedNode(NULL);
	else
		SetSelectedNode(pNodeGraphics->GetNode());

//	m_pTransferFunction->AddNode(new QNode(m_pTransferFunction, 50.0f, 30.0f, QColor(255, 160, 30, 255)));
}

void CTransferFunctionView::UpdateCanvas(void)
{
	m_EditRect = rect();
	m_EditRect.adjust(m_Margin, m_Margin, -m_Margin, -m_Margin);

	// Update canvas
	m_pCanvas->setRect(m_EditRect);
}

void CTransferFunctionView::UpdateGrid(void)
{
	// Horizontal grid lines
	const float DeltaY = 0.1f * m_EditRect.height();

	// Remove old horizontal grid lines
	foreach(QGraphicsLineItem* pLine, m_GridLinesHorizontal)
		m_pGraphicsScene->removeItem(pLine);

	// Clear the edges list
	m_GridLinesHorizontal.clear();

	for (int i = 1; i < 10; i++)
	{
		QGraphicsLineItem* pLine = new QGraphicsLineItem(QLineF(QPointF(m_EditRect.left(), m_EditRect.top() + i * DeltaY), QPointF(m_EditRect.right(), m_EditRect.top() + i * DeltaY)));

		pLine->setPen(m_GridPenHorizontal);

		m_pGraphicsScene->addItem(pLine);
		m_GridLinesHorizontal.append(pLine);
	}
}

void CTransferFunctionView::UpdateHistogram(void)
{
}

void CTransferFunctionView::UpdateEdges(void)
{
	// Remove old edges
	foreach(QGraphicsLineItem* pLine, m_Edges)
		m_pGraphicsScene->removeItem(pLine);

	// Clear the edges list
	m_Edges.clear();

	for (int i = 1; i < m_Nodes.size(); i++)
	{
		QPointF PtFrom(m_Nodes[i - 1]->GetCenter());
		QPointF PtTo(m_Nodes[i]->GetCenter());

		QGraphicsLineItem* pLine = new QGraphicsLineItem(QLineF(PtFrom, PtTo));
		
		// Set the pen
		pLine->setPen(QPen(QColor(110, 110, 100), 0.5));

		// Ensure the item is drawn in the right order
		pLine->setZValue(800);

		m_Edges.append(pLine);
		m_pGraphicsScene->addItem(pLine);
	}
}

void CTransferFunctionView::UpdateNodes(void)
{
	m_EditRect = rect();

	// Set the gradient stops
	foreach(CNodeGraphics* pNodeGraphics, m_Nodes)
	{
		QPointF Center;

		Center.setX(rect().left() + rect().width() * pNodeGraphics->GetNode()->GetNormalizedX());
		Center.setY(rect().top() + rect().height() - (pNodeGraphics->GetNode()->GetOpacity() * rect().height()));
		
		pNodeGraphics->SetCenter(Center);

//		pNodeGraphics->setPos(Center);
	}

	m_pOutline->setRect(m_EditRect);
}

void CTransferFunctionView::UpdateGradient(void)
{
	m_LinearGradient.setStart(rect().left(), rect().top());
	m_LinearGradient.setFinalStop(rect().right(), rect().top());

	QGradientStops GradientStops;

	// Set the gradient stops
	foreach(QNode* pTransferFunctionNode, m_pTransferFunction->m_Nodes)
	{
		QColor Color = pTransferFunctionNode->GetColor();

		// Clamp node opacity to obtain valid alpha for display
		float Alpha = qMin(1.0f, qMax(0.0f, pTransferFunctionNode->GetOpacity()));

		Color.setAlphaF(0.2f);

		GradientStops.append(QGradientStop((pTransferFunctionNode->GetPosition() - m_pTransferFunction->m_RangeMin) / m_pTransferFunction->m_Range, Color));
	}

	m_LinearGradient.setStops(GradientStops);
}

void CTransferFunctionView::UpdatePolygon(void)
{
	QPolygonF Polygon;

	// Set the gradient stops
	for (int i = 0; i < m_Nodes.size(); i++)
	{
		CNodeGraphics* pNodeGraphics = m_Nodes[i];

		QPointF Center;

		Center.setX(m_EditRect.left() + pNodeGraphics->GetNode()->GetNormalizedX() * m_EditRect.width());
		Center.setY(m_EditRect.bottom() - pNodeGraphics->GetNode()->GetOpacity() * m_EditRect.height());

		if (pNodeGraphics == m_Nodes.front())
		{
			QPointF CenterCopy = Center;

			CenterCopy.setY(m_EditRect.bottom());

			Polygon.append(CenterCopy);
		}

		Polygon.append(Center);

		if (pNodeGraphics == m_Nodes.back())
		{
			QPointF CenterCopy = Center;

			CenterCopy.setY(m_EditRect.bottom());

			Polygon.append(CenterCopy);
		}
	}

	m_pPolygon->setPolygon(Polygon);
	m_pPolygon->setBrush(QBrush(m_LinearGradient));
	m_pPolygon->setPen(QPen(Qt::PenStyle::NoPen));
}

void CTransferFunctionView::Update(void)
{
	setSceneRect(rect());

	UpdateCanvas();
//	UpdateHistogram();
	UpdateNodes();
	UpdateEdges();
//	UpdateGradient();
//	UpdatePolygon();
}

void CTransferFunctionView::OnNodeAdd(QNode* pTransferFunctionNode)
{
	CNodeGraphics* pNodeGraphics = new CNodeGraphics(NULL, pTransferFunctionNode, this);
	
	// Ensure the item is drawn in the right order
	pNodeGraphics->setZValue(900);

	m_pGraphicsScene->addItem(pNodeGraphics);

	m_Nodes.append(pNodeGraphics);

	/*
	qSort(m_Nodes.begin(), m_Nodes.end(), NodeLessThan);

	// Clear the edges list
	m_Edges.clear();

	foreach (QGraphicsLineItem* pItem, m_Edges)
		delete pItem;

	for (int i = 1; i < m_Nodes.size(); i++)
	{
		QGraphicsLineItem* pLine = new QGraphicsLineItem();
	}
	*/
}

void CTransferFunctionView::OnNodeRemove(QNode* pTransferFunctionNode)
{
	m_Edges.clear();
}

void CTransferFunctionView::OnNodeMove(CNodeGraphics* pNodeGraphics)
{
	m_Edges.clear();

	QNode * pNode = pNodeGraphics->GetNode();

//	pNode->SetPosition(m_pTransferFunction->m_RangeMin + (float)pNodeGraphics->pos().x() * (m_pTransferFunction->m_Range / (float)rect().width()));

//	pNode->SetOpacity(pNodeGraphics->pos().y() / rect().height());
}

void CTransferFunctionView::SetSelectedNode(QNode* pSelectedNode)
{
	m_pTransferFunction->SetSelectedNode(pSelectedNode);
}

void CTransferFunctionView::OnNodeSelectionChanged(QNode* pNode)
{
	if (pNode)
	{
		// Deselect all nodes
		foreach (CNodeGraphics* pNode, m_Nodes)
			pNode->setSelected(false);

		// Obtain node index
		const int NodeIndex = m_pTransferFunction->GetNodeIndex(pNode);

		// Select the node
		if (NodeIndex >= 0)
		{
			m_Nodes[NodeIndex]->setSelected(true);
			m_Nodes[NodeIndex]->update();
		}
	}
	else
	{
	}
}

CGradientView::CGradientView(QWidget* pParent, QTransferFunction* pTransferFunction) :
	QGraphicsView(pParent),
	m_pGraphicsScene(NULL),
	m_pTransferFunction(pTransferFunction),
	m_CheckerSize(10, 10),
	m_pGradientRectangle(NULL),
	m_LinearGradient()
{
	// Dimensions
	setFixedHeight(20);

	// Styling
	setFrameShadow(QFrame::Shadow::Sunken);
	setFrameShape(NoFrame);

	// Never show scrollbars
	setVerticalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);
	setHorizontalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);

	// Status and tooltip
	setStatusTip("Transfer function gradient");
	setToolTip("Transfer function gradient");

	// Create scene and apply
	m_pGraphicsScene = new QGraphicsScene(this);
	setScene(m_pGraphicsScene);

	// Turn antialiasing on
	setRenderHint(QPainter::Antialiasing);

	m_pGradientRectangle = new QGraphicsRectItem();
	m_pGradientRectangle->setBrush(QBrush(Qt::gray));
	m_pGradientRectangle->setPen(QPen(Qt::PenStyle::NoPen));
	m_pGraphicsScene->addItem(m_pGradientRectangle);

	// Setup connections
	connect(m_pTransferFunction, SIGNAL(FunctionChanged()), this, SLOT(Update()));
}

void CGradientView::drawBackground(QPainter* pPainter, const QRectF& Rectangle)
{
	setSceneRect(rect());
	centerOn(Rectangle.center());

	const int NumX = ceilf(Rectangle.width() / m_CheckerSize.width());

	pPainter->setPen(QPen(Qt::PenStyle::NoPen));
	pPainter->setBrush(QBrush(Qt::gray));

	for (int i = 0; i < NumX; i++)
	{
		if (i % 2 == 0)
		{
			pPainter->drawRect(i * m_CheckerSize.width(), 0.0, m_CheckerSize.width(), m_CheckerSize.height());
		}
		else
		{
			pPainter->drawRect(i * m_CheckerSize.width(), m_CheckerSize.height(), m_CheckerSize.width(), m_CheckerSize.height());
		}
	}
	
	// Set the gradient stops
	foreach(QNode* pNode, m_pTransferFunction->m_Nodes)
	{
		pPainter->setPen(QPen(Qt::PenStyle::SolidLine));

		pPainter->drawLine(QPointF(pNode->GetNormalizedX() * rect().width(), 0.0f), QPointF(pNode->GetNormalizedX() * rect().width(), 20.0f));
	}
}

void CGradientView::resizeEvent(QResizeEvent* pResizeEvent)
{
	Update();
}

void CGradientView::Update(void)
{
	setSceneRect(rect());

	m_LinearGradient.setStart(rect().left(), rect().top());
	m_LinearGradient.setFinalStop(rect().right(), rect().top());

	QGradientStops GradientStops;

	// Set the gradient stops
	foreach(QNode* pTransferFunctionNode, m_pTransferFunction->m_Nodes)
	{
		QColor Color = pTransferFunctionNode->GetColor();

		// Clamp node opacity to obtain valid alpha for display
		float Alpha = qMin(1.0f, qMax(0.0f, pTransferFunctionNode->GetOpacity()));

		Color.setAlphaF(Alpha);

		GradientStops.append(QGradientStop((pTransferFunctionNode->GetPosition() - m_pTransferFunction->m_RangeMin) / m_pTransferFunction->m_Range, Color));
	}

	m_LinearGradient.setStops(GradientStops);

	m_pGradientRectangle->setRect(rect());
	m_pGradientRectangle->setBrush(QBrush(m_LinearGradient));
	m_pGradientRectangle->setPen(QPen(Qt::darkGray));
//	m_pGradientRectangle->setVisible(false);
}

void CGradientView::OnNodeAdd(QNode* pTransferFunctionNode)
{
}

void CGradientView::OnNodeRemove(QNode* pTransferFunctionNode)
{
}

CNodePropertiesWidget::CNodePropertiesWidget(QWidget* pParent, QTransferFunction* pTransferFunction) :
	QWidget(pParent),
	m_pTransferFunction(pTransferFunction),
	m_pMainLayout(NULL),
	m_pSelectionLabel(NULL),
	m_pSelectionLayout(NULL),
	m_pNodeSelectionComboBox(NULL),
	m_pPreviousNodePushButton(NULL),
	m_pNextNodePushButton(NULL),
	m_pPositionLabel(NULL),
	m_pPositionSlider(NULL),
	m_pPositionSpinBox(NULL),
	m_pOpacityLabel(NULL),
	m_pOpacitySlider(NULL),
	m_pOpacitySpinBox(NULL),
	m_pColorLabel(NULL),
	m_pColorComboBox(NULL),
	m_pRoughnessLabel(NULL),
	m_pRoughnessSlider(NULL),
	m_pRoughnessSpinBox(NULL)
{
	setFixedHeight(100);
	
	// Node properties layout
	m_pMainLayout = new QGridLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
	m_pMainLayout->setContentsMargins(0, 0, 0, 0);

	setLayout(m_pMainLayout);

	// Node selection
	m_pSelectionLabel = new QLabel("Selection");
	m_pSelectionLabel->setStatusTip("Node selection");
	m_pSelectionLabel->setToolTip("Node selection");
	m_pMainLayout->addWidget(m_pSelectionLabel, 0, 0);

	m_pSelectionLayout = new QGridLayout();
	m_pSelectionLayout->setAlignment(Qt::AlignTop);
	m_pSelectionLayout->setContentsMargins(0, 0, 0, 0);
	
	m_pMainLayout->addLayout(m_pSelectionLayout, 0, 1, 1, 2);

	m_pNodeSelectionComboBox = new QComboBox;
	m_pNodeSelectionComboBox->setStatusTip("Node selection");
	m_pNodeSelectionComboBox->setToolTip("Node selection");
	m_pSelectionLayout->addWidget(m_pNodeSelectionComboBox, 0, 0);

	connect(m_pNodeSelectionComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnNodeSelectionChanged(int)));

	m_pPreviousNodePushButton = new QPushButton("<<");
	m_pPreviousNodePushButton->setStatusTip("Select previous node");
	m_pPreviousNodePushButton->setToolTip("Select previous node");
	m_pPreviousNodePushButton->setFixedWidth(30);
	m_pPreviousNodePushButton->setFixedHeight(20);
	m_pPreviousNodePushButton->updateGeometry();
	m_pSelectionLayout->addWidget(m_pPreviousNodePushButton, 0, 1);

	connect(m_pPreviousNodePushButton, SIGNAL(pressed()), this, SLOT(OnPreviousNode()));

	m_pNextNodePushButton = new QPushButton(">>");
	m_pNextNodePushButton->setStatusTip("Select next node");
	m_pNextNodePushButton->setToolTip("Select next node");
	m_pNextNodePushButton->setFixedWidth(30);
	m_pNextNodePushButton->setFixedHeight(20);
	m_pSelectionLayout->addWidget(m_pNextNodePushButton, 0, 2);
	
	connect(m_pNextNodePushButton, SIGNAL(pressed()), this, SLOT(OnNextNode()));

	// Position
	m_pPositionLabel = new QLabel("Position");
	m_pPositionLabel->setStatusTip("Node position");
	m_pPositionLabel->setToolTip("Node position");
	m_pMainLayout->addWidget(m_pPositionLabel, 1, 0);

	m_pPositionSlider = new QSlider(Qt::Orientation::Horizontal);
	m_pPositionSlider->setStatusTip("Node position");
	m_pPositionSlider->setToolTip("Drag to change node position");
	m_pPositionSlider->setRange(0, 100);
	m_pMainLayout->addWidget(m_pPositionSlider, 1, 1);
	
	m_pPositionSpinBox = new QSpinBox;
	m_pPositionSpinBox->setStatusTip("Node position");
	m_pPositionSpinBox->setToolTip("Node position");
    m_pPositionSpinBox->setRange(0, 100);
	m_pMainLayout->addWidget(m_pPositionSpinBox, 1, 2);

	// Opacity
	m_pOpacityLabel = new QLabel("Opacity");
	m_pMainLayout->addWidget(m_pOpacityLabel, 2, 0);

	m_pOpacitySlider = new QSlider(Qt::Orientation::Horizontal);
	m_pOpacitySlider->setRange(0, 100);
	m_pMainLayout->addWidget(m_pOpacitySlider, 2, 1);
	
	m_pOpacitySpinBox = new QSpinBox;
    m_pOpacitySpinBox->setRange(0, 100);
	m_pMainLayout->addWidget(m_pOpacitySpinBox, 2, 2);
	
//	connect(m_pFocalDistanceSlider, SIGNAL(valueChanged(int)), m_pFocalDistanceSpinBox, SLOT(setValue(int)));
//	connect(m_pFocalDistanceSlider, SIGNAL(valueChanged(int)), this, SLOT(SetFocalDistance(int)));
//	connect(m_pFocalDistanceSpinBox, SIGNAL(valueChanged(int)), m_pFocalDistanceSlider, SLOT(setValue(int)));

	// Color
	m_pColorLabel = new QLabel("Color");
	m_pMainLayout->addWidget(m_pColorLabel, 3, 0);

	m_pColorComboBox = new QComboBox;
	m_pMainLayout->addWidget(m_pColorComboBox, 3, 1, 1, 2);

	/*
	// Roughness
	m_pRoughnessLabel = new QLabel("Roughness");
	m_pMainLayout->addWidget(m_pRoughnessLabel, 4, 0);

	m_pRoughnessSlider = new QSlider(Qt::Orientation::Horizontal);
	m_pRoughnessSlider->setRange(0, 100);
	m_pMainLayout->addWidget(m_pRoughnessSlider, 4, 1);
	
	m_pRoughnessSpinBox = new QSpinBox;
    m_pRoughnessSpinBox->setRange(0, 100);
	m_pMainLayout->addWidget(m_pRoughnessSpinBox, 4, 2);
	*/
	
	// Setup connections for position
	connect(m_pPositionSlider, SIGNAL(valueChanged(int)), m_pPositionSpinBox, SLOT(setValue(int)));
	connect(m_pPositionSpinBox, SIGNAL(valueChanged(int)), m_pPositionSlider, SLOT(setValue(int)));
	connect(m_pPositionSlider, SIGNAL(valueChanged(int)), this, SLOT(OnPositionChanged(int)));
	connect(m_pTransferFunction, SIGNAL(FunctionChanged()), this, SLOT(OnTransferFunctionChanged()));

	// Setup connections for opacity
	connect(m_pOpacitySlider, SIGNAL(valueChanged(int)), m_pOpacitySpinBox, SLOT(setValue(int)));
	connect(m_pOpacitySpinBox, SIGNAL(valueChanged(int)), m_pOpacitySlider, SLOT(setValue(int)));
	connect(m_pOpacitySlider, SIGNAL(valueChanged(int)), this, SLOT(OnOpacityChanged(int)));

	// Respond to changes in node selection
	connect(m_pTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));
	
	// Respond to addition and removal of nodes
	connect(m_pTransferFunction, SIGNAL(NodeAdd(QNode*)), this, SLOT(OnNodeAdd(QNode*)));
	connect(m_pTransferFunction, SIGNAL(NodeRemove(QNode*)), this, SLOT(OnNodeRemove(QNode*)));
}

void CNodePropertiesWidget::OnNodeSelectionChanged(QNode* pNode)
{
	if (!pNode)
	{
		setEnabled(false);
	}
	else
	{
		setEnabled(true);

		m_pOpacitySlider->setValue(100.0f * pNode->GetOpacity());

		// Restrict the node's position
		m_pPositionSlider->setRange(pNode->GetMinX(), pNode->GetMaxX());
		m_pPositionSpinBox->setRange(pNode->GetMinX(), pNode->GetMaxX());

		// Obtain current node index
		const int CurrentNodeIndex = m_pTransferFunction->GetNodeIndex(pNode);

		// Reflect node selection change in node selection combo box
		m_pNodeSelectionComboBox->setCurrentIndex(CurrentNodeIndex);

		// Compute whether to enable/disable buttons
		const bool EnablePrevious	= CurrentNodeIndex > 0;
		const bool EnableNext		= CurrentNodeIndex < m_pTransferFunction->m_Nodes.size() - 1;
		const bool EnablePosition	= m_pTransferFunction->m_Nodes.front() != pNode && m_pTransferFunction->m_Nodes.back() != pNode;

		// Selectively enable/disable previous/next buttons
		m_pPreviousNodePushButton->setEnabled(EnablePrevious);
		m_pNextNodePushButton->setEnabled(EnableNext);

		// Enable/disable position label, slider and spinbox
		m_pPositionLabel->setEnabled(EnablePosition);
		m_pPositionSlider->setEnabled(EnablePosition);
		m_pPositionSpinBox->setEnabled(EnablePosition);

		// Create tooltip strings
		QString PreviousToolTip = EnablePrevious ? "Select node " + QString::number(CurrentNodeIndex) : "No previous node";
		QString NextToolTip		= EnableNext ? "Select node " + QString::number(CurrentNodeIndex + 2) : "No next node";

		// Update push button tooltips
		m_pPreviousNodePushButton->setStatusTip(PreviousToolTip);
		m_pPreviousNodePushButton->setToolTip(PreviousToolTip);
		m_pNextNodePushButton->setStatusTip(NextToolTip);
		m_pNextNodePushButton->setToolTip(NextToolTip);
	}
}

void CNodePropertiesWidget::OnNodeSelectionChanged(const int& Index)
{
	m_pTransferFunction->SetSelectedNode(Index);
}

void CNodePropertiesWidget::OnPreviousNode(void)
{
	m_pTransferFunction->SelectPreviousNode();
}

void CNodePropertiesWidget::OnNextNode(void)
{
	m_pTransferFunction->SelectNextNode();
}

void CNodePropertiesWidget::OnTransferFunctionChanged(void)
{
	if (m_pTransferFunction->m_pSelectedNode)
		m_pPositionSlider->setValue(m_pTransferFunction->m_pSelectedNode->GetPosition());
}

void CNodePropertiesWidget::OnPositionChanged(const int& Position)
{
	if (m_pTransferFunction->m_pSelectedNode)
	{
		m_pTransferFunction->m_pSelectedNode->SetPosition(Position);
//		m_pPositionSlider->setValue(Position);
	}
}

void CNodePropertiesWidget::OnOpacityChanged(const int& Opacity)
{
	if (m_pTransferFunction->m_pSelectedNode)
		m_pTransferFunction->m_pSelectedNode->SetOpacity(0.01f * Opacity);
}

void CNodePropertiesWidget::OnColorChanged(const QColor& Color)
{
	m_pNodeSelectionComboBox->clear();

	for (int i = 0; i < m_pTransferFunction->m_Nodes.size(); i++)
		m_pNodeSelectionComboBox->addItem("Node " + QString::number(i + 1));
}

void CNodePropertiesWidget::OnNodeAdd(QNode* pNode)
{
	m_pNodeSelectionComboBox->clear();

	for (int i = 0; i < m_pTransferFunction->m_Nodes.size(); i++)
		m_pNodeSelectionComboBox->addItem("Node " + QString::number(i + 1));
}

void CNodePropertiesWidget::OnNodeRemove(QNode* pNode)
{
	/*
	if (m_pTransferFunction->m_pSelectedNode)
		m_pTransferFunction->m_pSelectedNode->SetOpacity(0.01f * Opacity);
	*/
}

CTransferFunctionWidget::CTransferFunctionWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_pMainLayout(NULL),
	m_pTransferFunction(NULL),
	m_pTransferFunctionView(NULL),
	m_pGradientView(NULL),
	m_pNodePropertiesWidget(NULL)
{
	setTitle("Transfer Function");
	setToolTip("Transfer function properties");

	// Create main layout
	m_pMainLayout = new QGridLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
	setLayout(m_pMainLayout);

	// Create Qt transfer function
	m_pTransferFunction = new QTransferFunction();

	// Transfer function view
	m_pTransferFunctionView = new CTransferFunctionView(this, m_pTransferFunction);
	m_pMainLayout->addWidget(m_pTransferFunctionView);

	// Gradient view
	m_pGradientView = new CGradientView(this, m_pTransferFunction);
	m_pMainLayout->addWidget(m_pGradientView);

	// Gradient view
	m_pNodePropertiesWidget = new CNodePropertiesWidget(this, m_pTransferFunction);
	m_pMainLayout->addWidget(m_pNodePropertiesWidget);

	m_pTransferFunction->AddNode(new QNode(m_pTransferFunction, 0.0f, 0.0f, QColor(255, 0, 0, 128), false));
	m_pTransferFunction->AddNode(new QNode(m_pTransferFunction, 70.0f, 0.5f, QColor(255, 160, 30, 255)));
	m_pTransferFunction->AddNode(new QNode(m_pTransferFunction, 100.0f, 0.1f, QColor(55, 160, 255, 255)));
	m_pTransferFunction->AddNode(new QNode(m_pTransferFunction, 255.0f, 1.0f, QColor(10, 255, 0, 128), false));
}