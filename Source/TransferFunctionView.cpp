
// Precompiled headers
#include "Stable.h"

#include "TransferFunction.h"
#include "TransferFunctionView.h"
#include "NodeItem.h"
#include "RenderThread.h"

QTFView::QTFView(QWidget* pParent /*= NULL*/) :
	QGraphicsView(pParent),
	m_Margin(25, 15, 10, 15),
	m_CanvasRectangle(),
	m_Scene(),
	m_Background(NULL),
	m_Grid(NULL),
	m_HistogramItem(NULL),
	m_TransferFunctionItem(NULL)
{
	// Styling
 	setFrameShadow(Sunken);
 	setFrameShape(NoFrame);

	// Never show scrollbars
	setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

	setBackgroundBrush(QBrush(QColor(240, 240, 240)));

	setScene(&m_Scene);

	// Turn anti-aliasing on
	setRenderHint(QPainter::Antialiasing);

	m_TransferFunctionItem.SetTransferFunction(&gTransferFunction);

	m_Background.setEnabled(false);
	m_Grid.setEnabled(false);
	m_HistogramItem.setEnabled(false);
	m_TransferFunctionItem.setEnabled(false);

	m_Scene.addItem(&m_Background);
	m_Scene.addItem(&m_Grid);
 	m_Scene.addItem(&m_HistogramItem);
	m_Scene.addItem(&m_TransferFunctionItem);

	m_Background.setZValue(0);
	m_Grid.setZValue(100);
	m_HistogramItem.setZValue(200);
	m_TransferFunctionItem.setZValue(300);

 	m_Background.translate(m_Margin.GetLeft(), m_Margin.GetTop());
	m_Grid.translate(m_Margin.GetLeft(), m_Margin.GetTop());
 	m_HistogramItem.translate(m_Margin.GetLeft(), m_Margin.GetTop());
	m_TransferFunctionItem.translate(m_Margin.GetLeft(), m_Margin.GetTop());

	QObject::connect(&gTransferFunction, SIGNAL(Changed()), this, SLOT(OnTransferFunctionChanged()));
	QObject::connect(&gTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));
}

void QTFView::resizeEvent(QResizeEvent* pResizeEvent)
{
	QGraphicsView::resizeEvent(pResizeEvent);

	m_CanvasRectangle = rect();

	m_CanvasRectangle.adjust(m_Margin.GetLeft(),  m_Margin.GetTop(), -m_Margin.GetRight(), -m_Margin.GetBottom());

	m_Background.setRect(m_CanvasRectangle);
	m_Grid.setRect(m_CanvasRectangle);
	m_HistogramItem.setRect(m_CanvasRectangle);
	m_TransferFunctionItem.setRect(m_CanvasRectangle);

	m_HistogramItem.Update();
	m_TransferFunctionItem.Update();

	setSceneRect(rect());
}

void QTFView::SetHistogram(QHistogram& Histogram)
{
	m_HistogramItem.SetHistogram(Histogram);
}

void QTFView::OnTransferFunctionChanged(void)
{
	m_TransferFunctionItem.Update();
}

void QTFView::OnNodeSelectionChanged(QNode* pNode)
{
	// Deselect all nodes
	for (int i = 0; i < m_TransferFunctionItem.m_Nodes.size(); i++)
		m_TransferFunctionItem.m_Nodes[i]->setSelected(false);

	if (pNode)
	{
		for (int i = 0; i < m_TransferFunctionItem.m_Nodes.size(); i++)
		{
			if (m_TransferFunctionItem.m_Nodes[i]->m_pNode->GetID() == pNode->GetID())
				m_TransferFunctionItem.m_Nodes[i]->setSelected(true);
		}
	}
}

void QTFView::setEnabled(bool Enabled)
{
	QGraphicsView::setEnabled(Enabled);

	m_Background.setEnabled(Enabled);
	m_Grid.setEnabled(Enabled);
	m_HistogramItem.setEnabled(Enabled);
	m_TransferFunctionItem.setEnabled(Enabled);
}

void QTFView::mousePressEvent(QMouseEvent* pEvent)
{
	QGraphicsView::mousePressEvent(pEvent);

	// Get node item under cursor
	QNodeItem* pNodeItem = dynamic_cast<QNodeItem*>(scene()->itemAt(pEvent->posF() - QPointF(rect().left(), rect().top())));

	if (!pNodeItem)
	{
		// Add a new node if the user clicked the left button
		if (pEvent->button() == Qt::LeftButton && m_CanvasRectangle.contains(pEvent->pos()))
		{
			// Convert picked position to transfer function coordinates
			QPointF TransferFunctionPoint((pEvent->posF().x() - m_Margin.GetLeft()) / m_CanvasRectangle.width(), (pEvent->posF().y() - m_Margin.GetTop()) / m_CanvasRectangle.height());

			// Create new transfer function node
			QNode NewNode(&gTransferFunction, TransferFunctionPoint.x(), 1.0f - TransferFunctionPoint.y(), QColor::fromHsl((int)(((float)rand() / (float)RAND_MAX) * 359.0f), 150, 150), QColor::fromHsl(0, 0, 50), QColor::fromHsl(0, 0, 0));

			// Add to node list
			gTransferFunction.AddNode(NewNode);

			// Redraw
			m_TransferFunctionItem.Update();
		}

		if (pEvent->button() == Qt::RightButton)
		{
			// Other wise no node selection
			gTransferFunction.SetSelectedNode((QNode*)NULL);
		}
	}
	else
	{
		if (pEvent->button() == Qt::LeftButton)
		{
			gTransferFunction.SetSelectedNode(pNodeItem->m_pNode);
		}
		else if (pEvent->button() == Qt::RightButton)
		{
			const int Index = gTransferFunction.GetNodes().indexOf(*pNodeItem->m_pNode);

			// Remove transfer function node if not the first or last node
			if (Index != 0 && Index != gTransferFunction.GetNodes().size() - 1)
				gTransferFunction.RemoveNode(pNodeItem->m_pNode);
		}
	}
}