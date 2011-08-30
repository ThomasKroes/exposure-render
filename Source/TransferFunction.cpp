
#include "TransferFunction.h"

QTransferFunction gTransferFunction;

// Compare two transfer function nodes by position
bool CompareNodes(QNode NodeA, QNode NodeB)
{
	return NodeA.GetIntensity() < NodeB.GetIntensity();
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
	m_MaxY(1.0f),
	m_ID(0)
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
	emit IntensityChanged(this);
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

int QNode::GetID(void) const
{
	return m_ID;
}

void QNode::ReadXML(QDomElement& Parent)
{
	// Intensity
	const float NormalizedIntensity = Parent.firstChildElement("NormalizedIntensity").attribute("Value").toFloat();
	m_Intensity = m_pTransferFunction->m_RangeMin + (m_pTransferFunction->m_Range * NormalizedIntensity);

	// Opacity
	const float NormalizedOpacity = Parent.firstChildElement("NormalizedOpacity").attribute("Value").toFloat();
	m_Opacity = NormalizedOpacity;
	
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
	const float NormalizedIntensity = GetNormalizedIntensity();
	Intensity.setAttribute("Value", NormalizedIntensity);
	Node.appendChild(Intensity);

	// Opacity
	QDomElement Opacity = DOM.createElement("NormalizedOpacity");
	const float NormalizedOpacity = GetNormalizedOpacity();
	Opacity.setAttribute("Value", NormalizedOpacity);
	Node.appendChild(Opacity);

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
}

QTransferFunction& QTransferFunction::operator = (const QTransferFunction& Other)			
{
	blockSignals(true);

	m_Name			= Other.m_Name;
	m_Nodes			= Other.m_Nodes;
	m_RangeMin		= Other.m_RangeMin;
	m_RangeMax		= Other.m_RangeMax;
	m_Range			= Other.m_Range;
	m_pSelectedNode	= Other.m_pSelectedNode;
	m_Histogram		= Other.m_Histogram;

	for (int i = 0; i < m_Nodes.size(); i++)
	{
		// Notify us when the node changes
		connect(&m_Nodes[i], SIGNAL(NodeChanged(QNode*)), this, SLOT(OnNodeChanged(QNode*)));
	}

	// Allow events to be fired
	blockSignals(false);

	// Inform others that our node count has changed
	emit NodeCountChanged();

	// Update node's range
	UpdateNodeRanges();

	// Notify others that the function has changed selection has changed
	emit FunctionChanged();

	SetSelectedNode(NULL);

	return *this;
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
	m_RangeMin	= RangeMin;
	m_Range		= GetRange();
}

float QTransferFunction::GetRangeMax(void) const
{
	return m_RangeMax;
}

void QTransferFunction::SetRangeMax(const float& RangeMax)
{
	m_RangeMax	= RangeMax;
	m_Range		= GetRange();
}

float QTransferFunction::GetRange(void) const
{
	return m_RangeMax - m_RangeMin;
}

void QTransferFunction::SetSelectedNode(QNode* pSelectedNode)
{
	if (pSelectedNode == NULL)
		qDebug("Selection cleared");
	else
		qDebug("Node %d is being selected", pSelectedNode->GetID());

	m_pSelectedNode = pSelectedNode;
	emit SelectionChanged(m_pSelectedNode);
}

void QTransferFunction::SetSelectedNode(const int& Index)
{
	if (m_Nodes.size() <= 0)
		return;

	qDebug("Node %d is being selected", Index);

	// Compute new index
	const int NewIndex = qMin(m_Nodes.size(), qMax(0, Index));

	// Set selected node
	m_pSelectedNode = &m_Nodes[NewIndex];

	// Notify others that our selection has changed
	emit SelectionChanged(m_pSelectedNode);
}

QNode* QTransferFunction::GetSelectedNode(void)
{
	return m_pSelectedNode;
}

void QTransferFunction::OnNodeChanged(QNode* pNode)
{
	// Update node's range
	UpdateNodeRanges();

	emit FunctionChanged();
}

void QTransferFunction::SelectPreviousNode(void)
{
	if (!m_pSelectedNode)
		return;

	int Index = m_Nodes.indexOf(*GetSelectedNode());

	if (Index < 0)
		return;

	// Compute new index
	const int NewIndex = qMin(m_Nodes.size() - 1, qMax(0, Index - 1));

	// Set selected node
	SetSelectedNode(&m_Nodes[NewIndex]);
}

void QTransferFunction::SelectNextNode(void)
{
	if (!m_pSelectedNode)
		return;

	int Index = m_Nodes.indexOf(*GetSelectedNode());

	if (Index < 0)
		return;

	// Compute new index
	const int NewIndex = qMin(m_Nodes.size() - 1, qMax(0, Index + 1));

	// Set selected node
	SetSelectedNode(&m_Nodes[NewIndex]);
}

int	QTransferFunction::GetNodeIndex(QNode* pNode)
{
	if (pNode == NULL)
		return -1;
	
	return m_Nodes.indexOf(*pNode);
}

void QTransferFunction::AddNode(const float& Position, const float& Opacity, const QColor& Color)
{
	AddNode(QNode(this, Position, Opacity, Color));
}

void QTransferFunction::AddNode(const QNode& Node)
{
	// Inform others that our node count is about to change
	emit NodeCountChange();

	// Add the node to the list
	m_Nodes.append(Node);

	// Cache node
	QNode& CacheNode = m_Nodes.back();

	// Sort the transfer function nodes based on intensity
	qSort(m_Nodes.begin(), m_Nodes.end(), CompareNodes);

	// Update ID's
	for (int i = 0; i < m_Nodes.size(); i++)
		m_Nodes[i].m_ID = i;

	// Notify us when the node changes
	connect(&CacheNode, SIGNAL(NodeChanged(QNode*)), this, SLOT(OnNodeChanged(QNode*)));

	// Inform others that the transfer function has changed
	emit FunctionChanged();

	// Inform others that our node count has changed
	emit NodeCountChanged();

	// Compute node index
	const int NodeIndex = m_Nodes.indexOf(CacheNode);

	// Select the last node that was added
	SetSelectedNode(NodeIndex);

	qDebug("Added a node");
}

void QTransferFunction::RemoveNode(QNode* pNode)
{
	// Inform others that our node count is about to change
	emit NodeCountChange();

	// Remove the connection
	disconnect(pNode, SIGNAL(NodeChanged(QNode*)), this, SLOT(OnNodeChanged(QNode*)));

	// Remove from list and memory
	m_Nodes.remove(*pNode);

	// Update ID's
	for (int i = 0; i < m_Nodes.size(); i++)
		m_Nodes[i].m_ID = i;

	// Update node's range
	UpdateNodeRanges();

	// Inform others that the transfer function has changed
	emit FunctionChanged();

	// Inform others that our node count has changed
	emit NodeCountChanged();

	qDebug("Removed a node");
}

void QTransferFunction::UpdateNodeRanges(void)
{
	// Compute the node ranges
	for (int i = 0; i < m_Nodes.size(); i++)
	{
		QNode& Node = m_Nodes[i];

		if (i == 0)
		{
			Node.SetMinX(0.0f);
			Node.SetMaxX(0.0f);
		}
		else if (i == (m_Nodes.size() - 1))
		{
			Node.SetMinX(m_RangeMax);
			Node.SetMaxX(m_RangeMax);
		}
		else
		{
			QNode& NodeLeft		= m_Nodes[i - 1];
			QNode& NodeRight	= m_Nodes[i + 1];

			Node.SetMinX(NodeLeft.GetIntensity());
			Node.SetMaxX(NodeRight.GetIntensity());
		}
	}

//	qDebug("Updated node ranges");
}

const QNodeList& QTransferFunction::GetNodes(void) const
{
	return m_Nodes;
}

QNode& QTransferFunction::GetNode(const int& Index)
{
	return m_Nodes[Index];
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

	qDebug("Histogram was set");
}

void QTransferFunction::ReadXML(QDomElement& Parent)
{
	// Don't fire events during loading
	blockSignals(true);

	SetName(Parent.attribute("Name", "Failed"));
	SetRangeMin(Parent.attribute("RangeMin", "Failed").toInt());
	SetRangeMax(Parent.attribute("RangeMax", "Failed").toInt());

	QDomElement Nodes = Parent.firstChild().toElement();

	// Read child nodes
	for (QDomNode DomNode = Nodes.firstChild(); !DomNode.isNull(); DomNode = DomNode.nextSibling())
	{
		// Create new node
		QNode Node(this);

		// Load preset into it
		Node.ReadXML(DomNode.toElement());

		// Add the node to the list
		AddNode(Node);
	}

	// Allow events again
	blockSignals(false);

	UpdateNodeRanges();

	qDebug() << m_Name << "transfer function preset loaded";

	// Inform others that the transfer function has changed
	emit FunctionChanged();
}

void QTransferFunction::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Create transfer function preset root element
	QDomElement Root = DOM.createElement("Preset");
	Parent.appendChild(Root);

	Root.setAttribute("Name", m_Name);
	Root.setAttribute("RangeMin", m_RangeMin);
	Root.setAttribute("RangeMax", m_RangeMax);

	// Nodes
	QDomElement Nodes = DOM.createElement("Nodes");
	Root.appendChild(Nodes);

	for (int i = 0; i < m_Nodes.size(); i++)
	{
		m_Nodes[i].WriteXML(DOM, Nodes);
	}

	qDebug() << m_Name << "transfer function preset saved";
}


