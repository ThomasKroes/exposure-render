
#include "TransferFunction.h"

QTransferFunction gTransferFunction;

// Compare two transfer function nodes by position
bool CompareNodes(QNode* pNodeA, QNode* pNodeB)
{
	return pNodeA->GetIntensity() < pNodeB->GetIntensity();
}

QNode::QNode(QTransferFunction* pTransferFunction, const float& Intensity, const float& Opacity, const QColor& Color) :
	QObject(pTransferFunction),
	m_pTransferFunction(pTransferFunction),
	m_Intensity(Intensity),
	m_Opacity(Opacity),
	m_Color(Color),
	m_MinX(0.0f),
	m_MaxX(0.0f),
	m_MinY(0.0f),
	m_MaxY(1.0f)
{
}

float QNode::GetNormalizedIntensity(void) const 
{
	return (GetIntensity() - m_pTransferFunction->m_RangeMin) / m_pTransferFunction->m_Range;
}

void QNode::SetNormalizedIntensity(const float& NormalizedIntensity)
{
	SetIntensity(m_pTransferFunction->m_RangeMin + (m_pTransferFunction->m_Range * NormalizedIntensity));
}

float QNode::GetNormalizedOpacity(void) const 
{
	return GetOpacity();
}

void QNode::SetNormalizedOpacity(const float& NormalizedOpacity)
{
	SetOpacity(NormalizedOpacity);
}

float QNode::GetIntensity(void) const
{
	return m_Intensity;
}

void QNode::SetIntensity(const float& Position)
{
	m_Intensity = qMin(m_MaxX, qMax(Position, m_MinX));
	
	emit NodeChanged(this);
	emit PositionChanged(this);
}

float QNode::GetOpacity(void) const
{
	return m_Opacity;
}

void QNode::SetOpacity(const float& Opacity)
{
	m_Opacity = qMin(m_MaxY, qMax(Opacity, m_MinY));
	m_Opacity = Opacity;

	emit NodeChanged(this);
	emit OpacityChanged(this);
}

QColor QNode::GetColor(void) const
{
	return m_Color;
}

void QNode::SetColor(const QColor& Color)
{
	m_Color = Color;
	
	emit ColorChanged(this);
}

float QNode::GetMinX(void) const
{
	return m_MinX;
}

void QNode::SetMinX(const float& MinX)
{
	m_MinX = MinX;

	emit RangeChanged(this);
}

float QNode::GetMaxX(void) const
{
	return m_MaxX;
}

void QNode::SetMaxX(const float& MaxX)
{
	m_MaxX = MaxX;

	emit RangeChanged(this);
}

float QNode::GetMinY(void) const
{
	return m_MinY;
}

void QNode::SetMinY(const float& MinY)
{
	m_MinY = MinY;

	emit RangeChanged(this);
}

float QNode::GetMaxY(void) const
{
	return m_MaxY;
}

void QNode::SetMaxY(const float& MaxY)
{
	m_MaxY = MaxY;

	emit RangeChanged(this);
}

bool QNode::InRange(const QPointF& Point)
{
	return Point.x() >= m_MinX && Point.x() <= m_MaxX && Point.y() >= m_MinY && Point.y() <= m_MaxY;
}

void QNode::ReadXML(QDomElement& Parent)
{
	const float NormalizedIntensity = Parent.firstChildElement("NormalizedIntensity").nodeValue().toFloat();

	// Set intensity from normalized intensity
	SetNormalizedIntensity(NormalizedIntensity);
	
	QDomElement Kd = Parent.firstChildElement("Kd");

	m_Color.setRed(Kd.attribute("R").toInt());
	m_Color.setGreen(Kd.attribute("G").toInt());
	m_Color.setBlue(Kd.attribute("B").toInt());
}

void QNode::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Node
	QDomElement Node = DOM.createElement("Node");
	Parent.appendChild(Node);

	// Intensity
	QDomElement Intensity = DOM.createElement("NormalizedIntensity");
	Intensity.setAttribute("Value", GetNormalizedIntensity());
	Node.appendChild(Intensity);

	// Kd
	QDomElement Kd = DOM.createElement("Kd");
	Kd.setAttribute("R", m_Color.red());
	Kd.setAttribute("G", m_Color.green());
	Kd.setAttribute("B", m_Color.blue());
	Node.appendChild(Kd);

	// Kt
	QDomElement Kt = DOM.createElement("Kt");
	Kt.setAttribute("R", m_Color.red());
	Kt.setAttribute("G", m_Color.green());
	Kt.setAttribute("B", m_Color.blue());
	Node.appendChild(Kt);

	// Ks
	QDomElement Ks = DOM.createElement("Ks");
	Ks.setAttribute("R", m_Color.red());
	Ks.setAttribute("G", m_Color.green());
	Ks.setAttribute("B", m_Color.blue());
	Node.appendChild(Ks);
}

QTransferFunction::QTransferFunction(QObject* pParent, const QString& Name) :
	QObject(pParent),
	m_Name(Name),
	m_Nodes(),
	m_RangeMin(0.0f),
	m_RangeMax(255.0f),
	m_Range(m_RangeMax - m_RangeMin),
	m_pSelectedNode(NULL),
	m_Histogram()
{
	blockSignals(true);

	AddNode(0.0f, 0.0f, Qt::black);
	AddNode(1.0f, 1.0f, Qt::white);

	blockSignals(false);
}

QString	QTransferFunction::GetName(void) const
{
	return m_Name;
}

void QTransferFunction::SetName(const QString& Name)
{
	m_Name = Name;
}

float QTransferFunction::GetRangeMin(void) const
{
	return m_RangeMin;
}

void QTransferFunction::SetRangeMin(const float& RangeMin)
{
	m_RangeMin = RangeMin;
	UpdateNodeRanges();
}

float QTransferFunction::GetRangeMax(void) const
{
	return m_RangeMax;
}

void QTransferFunction::SetRangeMax(const float& RangeMax)
{
	m_RangeMax = RangeMax;
	UpdateNodeRanges();
}

void QTransferFunction::SetSelectedNode(QNode* pSelectedNode)
{
	m_pSelectedNode = pSelectedNode;
	emit SelectionChanged(m_pSelectedNode);
}

void QTransferFunction::SetSelectedNode(const int& Index)
{
	if (m_Nodes.size() <= 0)
		return;

	// Compute new index
	const int NewIndex = qMin(m_Nodes.size(), qMax(0, Index));

	// Set selected node
	m_pSelectedNode = m_Nodes[NewIndex];

	// Notify others that our selection has changed
	emit SelectionChanged(m_pSelectedNode);
}

const QNode* QTransferFunction::GetSelectedNode(void)
{
	return m_pSelectedNode;
}

void QTransferFunction::OnNodeChanged(QNode* pNode)
{
	UpdateNodeRanges();

	emit FunctionChanged();
}

void QTransferFunction::SelectPreviousNode(void)
{
	if (!m_pSelectedNode)
		return;

	int Index = GetNodeIndex(m_pSelectedNode);

	if (Index < 0)
		return;

	// Compute new index
	const int NewIndex = qMin(m_Nodes.size() - 1, qMax(0, Index - 1));

	// Set selected node
	SetSelectedNode(m_Nodes[NewIndex]);
}

void QTransferFunction::SelectNextNode(void)
{
	if (!m_pSelectedNode)
		return;

	int Index = GetNodeIndex(m_pSelectedNode);

	if (Index < 0)
		return;

	// Compute new index
	const int NewIndex = qMin(m_Nodes.size() - 1, qMax(0, Index + 1));

	// Set selected node
	SetSelectedNode(m_Nodes[NewIndex]);
}

int	QTransferFunction::GetNodeIndex(QNode* pNode)
{
	if (pNode == NULL)
		return -1;

	return m_Nodes.indexOf(pNode);
}

void QTransferFunction::AddNode(const float& Position, const float& Opacity, const QColor& Color)
{
	AddNode(new QNode(this, Position, Opacity, Color));
}

void QTransferFunction::AddNode(QNode* pNode)
{
	m_Nodes.append(pNode);

	// Sort the transfer function nodes
	qSort(gTransferFunction.m_Nodes.begin(), gTransferFunction.m_Nodes.end(), CompareNodes);

	// Update node's range
	UpdateNodeRanges();

	// Emit
	emit NodeAdd(m_Nodes.back());
	emit FunctionChanged();

	// Notify us when the node changes
	connect(pNode, SIGNAL(NodeChanged(QNode*)), this, SLOT(OnNodeChanged(QNode*)));
}

void QTransferFunction::RemoveNode(QNode* pNode)
{
	// Let others know that we are about to remove a node
	emit NodeRemove(pNode);

	// Remove the connection
	disconnect(pNode, SIGNAL(NodeChanged(QNode*)), this, SLOT(OnNodeChanged(QNode*)));

	// Remove from list and memory
	m_Nodes.remove(m_Nodes.indexOf(pNode));
	delete pNode;

	// Update node's range
	UpdateNodeRanges();

	// Let others know that we remove a node, and that the transfer function has changed
	emit NodeRemoved(pNode);

	// Emit
	emit FunctionChanged();
}

void QTransferFunction::UpdateNodeRanges(void)
{
	// Compute the node ranges
	for (int i = 0; i < m_Nodes.size(); i++)
	{
		QNode* pNode = gTransferFunction.m_Nodes[i];

		if (pNode == gTransferFunction.m_Nodes.front())
		{
			pNode->SetMinX(0.0f);
			pNode->SetMaxX(0.0f);
		}
		else if (pNode == gTransferFunction.m_Nodes.back())
		{
			pNode->SetMinX(gTransferFunction.m_RangeMax);
			pNode->SetMaxX(gTransferFunction.m_RangeMax);
		}
		else
		{
			QNode* pNodeLeft	= gTransferFunction.m_Nodes[i - 1];
			QNode* pNodeRight	= gTransferFunction.m_Nodes[i + 1];

			pNode->SetMinX(pNodeLeft->GetIntensity());
			pNode->SetMaxX(pNodeRight->GetIntensity());
		}
	}
}

const QVector<QNode*>& QTransferFunction::GetNodes(void) const
{
	return m_Nodes;
}

const QHistogram& QTransferFunction::GetHistogram(void) const
{
	return m_Histogram;
}

void QTransferFunction::SetHistogram(const int* pBins, const int& NoBins)
{
	m_Histogram.m_Bins.clear();

	for (int i = 0; i < NoBins; i++)
	{
		if (pBins[i] > m_Histogram.m_Max)
			m_Histogram.m_Max = pBins[i];
	}

	for (int i = 0; i < NoBins; i++)
		m_Histogram.m_Bins.append(pBins[i]);

	// Inform other that the histogram has changed
	emit HistogramChanged();
}

void QTransferFunction::ReadXML(QDomElement& Parent)
{
	m_Name		= Parent.attribute("Name", "Failed");
	m_RangeMin	= Parent.attribute("RangeMin", "Failed").toFloat();
	m_RangeMax	= Parent.attribute("RangeMax", "Failed").toFloat();
	m_Range		= Parent.attribute("Range", "Failed").toFloat();

	// Read child nodes
	for (int i = 0; i < Parent.childNodes().count(); i++)
	{
		// Create new node
		QNode Node(this);

		// Load preset into it
		Node.ReadXML(Parent.childNodes().at(i).toElement());
	}
}

void QTransferFunction::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Create transfer function preset root element
	QDomElement Root = DOM.createElement("Preset");
	Parent.appendChild(Root);

	Root.setAttribute("Name", m_Name);
	Root.setAttribute("RangeMin", m_RangeMin);
	Root.setAttribute("RangeMax", m_RangeMax);
	Root.setAttribute("Range", m_Range);

	// Nodes
	QDomElement Nodes = DOM.createElement("Nodes");
	Root.appendChild(Nodes);

	foreach (QNode* pNode, m_Nodes)
	{
		pNode->WriteXML(DOM, Nodes);
	}
}




