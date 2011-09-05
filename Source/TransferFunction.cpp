
#include "TransferFunction.h"

QTransferFunction gTransferFunction;

// Compare two transfer function nodes by position
bool CompareNodes(QNode NodeA, QNode NodeB)
{
	return NodeA.GetIntensity() < NodeB.GetIntensity();
}

QNode::QNode(QTransferFunction* pTransferFunction, const float& Intensity, const float& Opacity, const QColor& DiffuseColor, const QColor& SpecularColor, const float& Roughness) :
	QPresetXML(pTransferFunction),
	m_pTransferFunction(pTransferFunction),
	m_Intensity(Intensity),
	m_Opacity(Opacity),
	m_DiffuseColor(DiffuseColor),
	m_SpecularColor(SpecularColor),
	m_Roughness(Roughness),
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

void QNode::SetIntensity(const float& Intensity)
{
	m_Intensity = qMin(m_MaxX, qMax(Intensity, m_MinX));
	
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

QColor QNode::GetDiffuseColor(void) const
{
	return m_DiffuseColor;
}

void QNode::SetDiffuseColor(const QColor& DiffuseColor)
{
	m_DiffuseColor = DiffuseColor;
	
	emit NodeChanged(this);
	emit DiffuseColorChanged(this);
}

QColor QNode::GetSpecularColor(void) const
{
	return m_SpecularColor;
}

void QNode::SetSpecularColor(const QColor& SpecularColor)
{
	m_SpecularColor = SpecularColor;

	emit NodeChanged(this);
	emit SpecularColorChanged(this);
}

float QNode::GetRoughness(void) const
{
	return m_Roughness;
}

void QNode::SetRoughness(const float& Roughness)
{
	m_Roughness = Roughness;

	emit NodeChanged(this);
	emit RoughnessChanged(this);
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
	QPresetXML::ReadXML(Parent);

	// Intensity
	const float NormalizedIntensity = Parent.firstChildElement("NormalizedIntensity").attribute("Value").toFloat();
	m_Intensity = m_pTransferFunction->m_RangeMin + (m_pTransferFunction->m_Range * NormalizedIntensity);

	// Opacity
	const float NormalizedOpacity = Parent.firstChildElement("NormalizedOpacity").attribute("Value").toFloat();
	m_Opacity = NormalizedOpacity;
	
	// Diffuse Color
	QDomElement DiffuseColor = Parent.firstChildElement("DiffuseColor");
	m_DiffuseColor.setRed(DiffuseColor.attribute("R").toInt());
	m_DiffuseColor.setGreen(DiffuseColor.attribute("G").toInt());
	m_DiffuseColor.setBlue(DiffuseColor.attribute("B").toInt());

	// Specular Color
	QDomElement SpecularColor = Parent.firstChildElement("SpecularColor");
	m_SpecularColor.setRed(SpecularColor.attribute("R").toInt());
	m_SpecularColor.setGreen(SpecularColor.attribute("G").toInt());
	m_SpecularColor.setBlue(SpecularColor.attribute("B").toInt());

	// Roughness
	m_Roughness = Parent.firstChildElement("Roughness").attribute("Value").toFloat();
}

QDomElement QNode::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Node
	QDomElement Node = DOM.createElement("Node");
	Parent.appendChild(Node);

	QPresetXML::WriteXML(DOM, Node);

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

	// Diffuse Color
	QDomElement DiffuseColor = DOM.createElement("DiffuseColor");
	DiffuseColor.setAttribute("R", m_DiffuseColor.red());
	DiffuseColor.setAttribute("G", m_DiffuseColor.green());
	DiffuseColor.setAttribute("B", m_DiffuseColor.blue());
	Node.appendChild(DiffuseColor);

	// Specular Color
	QDomElement SpecularColor = DOM.createElement("SpecularColor");
	SpecularColor.setAttribute("R", m_SpecularColor.red());
	SpecularColor.setAttribute("G", m_SpecularColor.green());
	SpecularColor.setAttribute("B", m_SpecularColor.blue());
	Node.appendChild(SpecularColor);

	// Roughness
	QDomElement Roughness = DOM.createElement("Roughness");
	Roughness.setAttribute("Value", m_Roughness);
	Node.appendChild(Roughness);

	return Node;
}

QTransferFunction::QTransferFunction(QObject* pParent, const QString& Name) :
	QPresetXML(pParent),
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
	QPresetXML::operator=(Other);

	blockSignals(true);

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

void QTransferFunction::OnNodeChanged(QNode* pNode)
{
	// Update node's range
	UpdateNodeRanges();

	emit FunctionChanged();
}

void QTransferFunction::SetSelectedNode(QNode* pSelectedNode)
{
	m_pSelectedNode = pSelectedNode;
	emit SelectionChanged(m_pSelectedNode);
}

void QTransferFunction::SetSelectedNode(const int& Index)
{
	if (m_Nodes.size() <= 0)
	{
		m_pSelectedNode = NULL;
	}
	else
	{
		// Compute new index
		const int NewIndex = qMin(m_Nodes.size() - 1, qMax(0, Index));

		// Set selected node
		m_pSelectedNode = &m_Nodes[NewIndex];
	}

	// Notify others that our selection has changed
	emit SelectionChanged(m_pSelectedNode);
}

QNode* QTransferFunction::GetSelectedNode(void)
{
	return m_pSelectedNode;
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

void QTransferFunction::AddNode(const float& Position, const float& Opacity, const QColor& DiffuseColor, const QColor& SpecularColor, const float& Roughness)
{
	AddNode(QNode(this, Position, Opacity, DiffuseColor, SpecularColor, Roughness));
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

	// Update ranges
	UpdateNodeRanges();

	// Notify us when the node changes
	connect(&CacheNode, SIGNAL(NodeChanged(QNode*)), this, SLOT(OnNodeChanged(QNode*)));

	for (int i = 0; i < m_Nodes.size(); i++)
	{
		if (Node.GetIntensity() == m_Nodes[i].GetIntensity())
			SetSelectedNode(&m_Nodes[i]);
	}

	// Inform others that the transfer function has changed
	emit FunctionChanged();

	// Inform others that our node count has changed
	emit NodeCountChanged();
}

void QTransferFunction::RemoveNode(QNode* pNode)
{
	if (!pNode)
		return;

	// Inform others that our node count is about to change
	emit NodeCountChange();

	// Remove the connection
	disconnect(pNode, SIGNAL(NodeChanged(QNode*)), this, SLOT(OnNodeChanged(QNode*)));

	// Node index of the to be removed node
	int NodeIndex = m_Nodes.indexOf(*pNode);

	// Remove from list and memory
	m_Nodes.remove(*pNode);

	// Update ID's
	for (int i = 0; i < m_Nodes.size(); i++)
		m_Nodes[i].m_ID = i;

	// Update node's range
	UpdateNodeRanges();

	// Select the previous node
	NodeIndex = qMax(0, NodeIndex - 1);

	SetSelectedNode(NodeIndex);

	// Inform others that the transfer function has changed
	emit FunctionChanged();

	// Inform others that our node count has changed
	emit NodeCountChanged();
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
}

void QTransferFunction::ReadXML(QDomElement& Parent)
{
	QPresetXML::ReadXML(Parent);

	// Don't fire events during loading
	blockSignals(true);

//	SetRangeMin(Parent.attribute("RangeMin", "Failed").toInt());
//	SetRangeMax(Parent.attribute("RangeMax", "Failed").toInt());

	QDomElement Nodes = Parent.firstChild().toElement();

	// Read child nodes
	for (QDomNode DomNode = Nodes.firstChild(); !DomNode.isNull(); DomNode = DomNode.nextSibling())
	{
		// Create new node
		QNode Light(this);

		// Load preset into it
		Light.ReadXML(DomNode.toElement());

		// Add the node to the list
		AddNode(Light);
	}

	// Allow events again
	blockSignals(false);

	UpdateNodeRanges();

	// Inform others that the transfer function has changed
	emit FunctionChanged();
}

QDomElement QTransferFunction::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Preset
	QDomElement Preset = DOM.createElement("Preset");
	Parent.appendChild(Preset);

	QPresetXML::WriteXML(DOM, Preset);

	Parent.appendChild(Preset);
	Preset.setAttribute("RangeMin", m_RangeMin);
	Preset.setAttribute("RangeMax", m_RangeMax);

	// Nodes
	QDomElement Nodes = DOM.createElement("Nodes");
	Preset.appendChild(Nodes);

	for (int i = 0; i < m_Nodes.size(); i++)
		m_Nodes[i].WriteXML(DOM, Nodes);

	return Preset;
}