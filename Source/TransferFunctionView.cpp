
#include "TransferFunction.h"
#include "TransferFunctionView.h"
#include "NodeItem.h"
#include "RenderThread.h"

QTransferFunctionView::QTransferFunctionView(QWidget* pParent) :
	QGraphicsView(pParent),
	m_GraphicsScene(),
	m_TransferFunctionCanvas(NULL, &m_GraphicsScene),
	m_TransferFunctionGradient(NULL, &m_GraphicsScene),
	m_MarginTop(8.0f),
	m_MarginBottom(42.0f),
	m_MarginLeft(15.0f),
	m_MarginRight(8.0f),
	m_AxisLabelX(NULL, ""),
	m_AxisLabelY(NULL, "")
{
	// Set the size policy, making sure the widget fits nicely in the layout
//	setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

	// Styling
	setFrameShadow(Sunken);
	setFrameShape(NoFrame);

	setStyleSheet("border-color: red;");

	// Never show scrollbars
	setVerticalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);
	setHorizontalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);

	// Status and tooltip
	setStatusTip("Transfer function editor");
	setToolTip("Transfer function editor");

	// Create scene and apply
	setScene(&m_GraphicsScene);

	// Turn anti-aliasing on
	setRenderHint(QPainter::Antialiasing);

	// Respond to changes in the transfer function
	connect(&gTransferFunction, SIGNAL(FunctionChanged()), this, SLOT(Update()));

	// Create the transfer function canvas and add it to the scene
	m_TransferFunctionCanvas.translate(m_MarginLeft, m_MarginTop);
	
	// Respond to changes in node selection
	connect(&gTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));
	connect(&gTransferFunction, SIGNAL(HistogramChanged(void)), this, SLOT(OnHistogramChanged(void)));

	// X-axis label
	m_AxisLabelX.m_Text =  "Density";
	m_GraphicsScene.addItem(&m_AxisLabelX);

	// Y-axis label
	m_AxisLabelY.m_Text = "Opacity";
	m_GraphicsScene.addItem(&m_AxisLabelY);
}

void QTransferFunctionView::drawBackground(QPainter* pPainter, const QRectF& Rectangle)
{
	QGraphicsView::drawBackground(pPainter, Rectangle);

	setBackgroundBrush(QBrush(QColor(240, 240, 240)));
}

void QTransferFunctionView::Update(void)
{
	m_TransferFunctionCanvas.Update();
	m_TransferFunctionGradient.Update();

	if (!Scene())
		return;

	Scene()->m_TransferFunctions.m_Opacity.m_NoNodes		= gTransferFunction.GetNodes().size();
	Scene()->m_TransferFunctions.m_DiffuseColor.m_NoNodes	= gTransferFunction.GetNodes().size();
	Scene()->m_TransferFunctions.m_SpecularColor.m_NoNodes	= gTransferFunction.GetNodes().size();
	Scene()->m_TransferFunctions.m_Roughness.m_NoNodes		= gTransferFunction.GetNodes().size();

	for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
	{
		QNode& Node = gTransferFunction.GetNode(i);

		// Positions
		Scene()->m_TransferFunctions.m_Opacity.m_P[i]			= Node.GetIntensity();
		Scene()->m_TransferFunctions.m_DiffuseColor.m_P[i]		= Node.GetIntensity();
		Scene()->m_TransferFunctions.m_SpecularColor.m_P[i]		= Node.GetIntensity();
		Scene()->m_TransferFunctions.m_Roughness.m_P[i]			= Node.GetIntensity();
		
		// Colors
		Scene()->m_TransferFunctions.m_Opacity.m_C[i]			= CColorRgbHdr(Node.GetOpacity());
		Scene()->m_TransferFunctions.m_DiffuseColor.m_C[i]		= CColorRgbHdr(Node.GetDiffuseColor().redF(), Node.GetDiffuseColor().greenF(), Node.GetDiffuseColor().blueF());
		Scene()->m_TransferFunctions.m_SpecularColor.m_C[i]		= CColorRgbHdr(Node.GetSpecularColor().redF(), Node.GetSpecularColor().greenF(), Node.GetSpecularColor().blueF());
		Scene()->m_TransferFunctions.m_Roughness.m_C[i]			= CColorRgbHdr(Node.GetRoughness());
	}

	Scene()->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

void QTransferFunctionView::OnNodeSelectionChanged(QNode* pNode)
{
	/*
	// Deselect all nodes
 	for (int i = 0; i < m_TransferFunctionCanvas.m_Nodes.size(); i++)
 		m_TransferFunctionCanvas.m_Nodes[i]->setSelected(false);

	if (pNode)
	{
		for (int i = 0; i < m_TransferFunctionCanvas.m_Nodes.size(); i++)
		{
			if (m_TransferFunctionCanvas.m_Nodes[i]->m_pNode->GetID() == pNode->GetID())
				m_TransferFunctionCanvas.m_Nodes[i]->setSelected(true);
		}
	}
	*/
}

void QTransferFunctionView::OnHistogramChanged(void)
{
	m_TransferFunctionCanvas.UpdateHistogram();
}

void QTransferFunctionView::resizeEvent(QResizeEvent* pResizeEvent)
{
	QGraphicsView::resizeEvent(pResizeEvent);

	setSceneRect(rect());

	QRectF CanvasRect = m_TransferFunctionCanvas.rect();

	CanvasRect.setWidth(rect().width() - m_MarginLeft - m_MarginRight);
	CanvasRect.setHeight(rect().height() - m_MarginTop - m_MarginBottom);

	m_TransferFunctionCanvas.setRect(CanvasRect);
	m_TransferFunctionCanvas.Update();
	m_TransferFunctionCanvas.UpdateGrid();
	m_TransferFunctionCanvas.UpdateHistogram();

	// Update transfer function gradient
	QRectF GradientRect = m_TransferFunctionCanvas.rect();

	GradientRect.setWidth(rect().width() - m_MarginLeft - m_MarginRight);
	GradientRect.setHeight(18);

	m_TransferFunctionGradient.setRect(GradientRect);
	m_TransferFunctionGradient.setPos(m_MarginLeft, CanvasRect.height() + 15);
	m_TransferFunctionGradient.Update();

	// Configure x-axis label
	m_AxisLabelX.setRect(QRectF(0, 0, CanvasRect.width(), 12));
	m_AxisLabelX.setX(m_MarginLeft);
	m_AxisLabelX.setY(m_MarginTop + CanvasRect.height() + 31);
	m_AxisLabelX.m_Text = "Intensity: [" + QString::number(gTransferFunction.GetRangeMin()) + ", " + QString::number(gTransferFunction.GetRangeMax()) + "]";

	// Configure y-axis label
	m_AxisLabelY.setRect(QRectF(0, 0, CanvasRect.height(), 9));
	m_AxisLabelY.setPos(0, m_MarginTop + CanvasRect.height());
	m_AxisLabelY.setRotation(-90.0f);
	m_AxisLabelY.m_Text = "Opacity (%): [0 - 100]";
}

void QTransferFunctionView::mousePressEvent(QMouseEvent* pEvent)
{
	QGraphicsView::mousePressEvent(pEvent);

	// Get node item under cursor
	QNodeItem* pNodeItem = dynamic_cast<QNodeItem*>(scene()->itemAt(pEvent->posF()));

	if (!pNodeItem)
	{
		// Add a new node if the user clicked the left button
		if (pEvent->button() == Qt::MouseButton::LeftButton && m_TransferFunctionCanvas.rect().contains(pEvent->posF() - QPointF(m_MarginLeft, m_MarginTop)))
		{
			// Convert picked position to transfer function coordinates
			QPointF TfPoint = m_TransferFunctionCanvas.SceneToTransferFunction(pEvent->posF() - QPointF(m_MarginLeft, m_MarginTop));

			// Generate random color
			int R = (int)(((float)rand() / (float)RAND_MAX) * 255.0f);
			int G = (int)(((float)rand() / (float)RAND_MAX) * 255.0f);
			int B = (int)(((float)rand() / (float)RAND_MAX) * 255.0f);

			// Create new transfer function node
			QNode NewNode(&gTransferFunction, TfPoint.x(), TfPoint.y(), QColor(R, G, B, 255));

			// Add to node list
			gTransferFunction.AddNode(NewNode);

			// Redraw
			m_TransferFunctionCanvas.Update();
		}

		if (pEvent->button() == Qt::MouseButton::RightButton)
		{
			// Other wise no node selection
			gTransferFunction.SetSelectedNode((QNode*)NULL);
		}
	}
	else
	{
		if (pEvent->button() == Qt::MouseButton::LeftButton)
		{
			gTransferFunction.SetSelectedNode(pNodeItem->m_pNode);
		}
		else if (pEvent->button() == Qt::MouseButton::RightButton)
		{
			const int Index = gTransferFunction.GetNodes().indexOf(*pNodeItem->m_pNode);

			// Remove transfer function node if not the first or last node
			if (Index != 0 && Index != gTransferFunction.GetNodes().size() - 1)
				gTransferFunction.RemoveNode(pNodeItem->m_pNode);
		}
	}
}