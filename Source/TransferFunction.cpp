
// Precompiled headers
#include "Stable.h"

#include "TransferFunction.h"

QTransferFunction gTransferFunction;

// Compare two transfer function nodes by intensity
bool CompareNodes(QNode NodeA, QNode NodeB)
{
	return NodeA.GetIntensity() < NodeB.GetIntensity();
}

QTransferFunction::QTransferFunction(QObject* pParent, const QString& Name) :
	QPresetXML(pParent),
	m_Nodes(),
	m_pSelectedNode(NULL)
{
}

QTransferFunction::QTransferFunction(const QTransferFunction& Other)
{
	*this = Other;
};

QTransferFunction& QTransferFunction::operator = (const QTransferFunction& Other)			
{
	QPresetXML::operator=(Other);

	blockSignals(true);
	
	m_Nodes			= Other.m_Nodes;
	m_pSelectedNode	= Other.m_pSelectedNode;

	// Notify us when the nodes change
	for (int i = 0; i < m_Nodes.size(); i++)
		connect(&m_Nodes[i], SIGNAL(NodeChanged(QNode*)), this, SLOT(OnNodeChanged(QNode*)));

	// Update node's range
	UpdateNodeRanges();

	blockSignals(false);

	// Notify others that the function has changed selection has changed
	emit FunctionChanged();

	SetSelectedNode(NULL);

	return *this;
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

void QTransferFunction::SelectFirstNode(void)
{
	if (m_Nodes.size() == 0)
		return;

	SetSelectedNode(&m_Nodes[0]);
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

void QTransferFunction::SelectLastNode(void)
{
	if (m_Nodes.size() == 0)
		return;

	SetSelectedNode(&m_Nodes[m_Nodes.size() - 1]);
}

int	QTransferFunction::GetNodeIndex(QNode* pNode)
{
	if (pNode == NULL)
		return -1;
	
	return m_Nodes.indexOf(*pNode);
}

void QTransferFunction::AddNode(const float& Intensity, const float& Opacity, const QColor& Diffuse, const QColor& Specular, const QColor& Emission, const float& Roughness)
{
	AddNode(QNode(this, Intensity, Opacity, Diffuse, Specular, Emission, Roughness));
}

void QTransferFunction::AddNode(const QNode& Node)
{
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
//	UpdateNodeRanges();

	// Notify us when the node changes
	connect(&CacheNode, SIGNAL(NodeChanged(QNode*)), this, SLOT(OnNodeChanged(QNode*)));

	for (int i = 0; i < m_Nodes.size(); i++)
	{
		if (Node.GetIntensity() == m_Nodes[i].GetIntensity())
			SetSelectedNode(&m_Nodes[i]);
	}

	// Inform others that the transfer function has changed
	emit FunctionChanged();

	Log("Inserted node", "layer-select-point");
}

void QTransferFunction::RemoveNode(QNode* pNode)
{
	if (!pNode)
		return;

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

	Log("Removed node", "layer-select-point");
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
			Node.SetMinX(1.0f);
			Node.SetMaxX(1.0f);
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

void QTransferFunction::ReadXML(QDomElement& Parent)
{
	QPresetXML::ReadXML(Parent);

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

	// Update node's range
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

	// Nodes
	QDomElement Nodes = DOM.createElement("Nodes");
	Preset.appendChild(Nodes);

	for (int i = 0; i < m_Nodes.size(); i++)
		m_Nodes[i].WriteXML(DOM, Nodes);

	return Preset;
}

QTransferFunction QTransferFunction::Default(void)
{
	QTransferFunction DefaultTransferFunction;

	DefaultTransferFunction.SetName("Default");
	DefaultTransferFunction.AddNode(0.0f, 0.0f, Qt::gray, QColor(10, 10, 10), Qt::black, 100.0f);
	DefaultTransferFunction.AddNode(0.3f, 0.0f, Qt::gray, QColor(10, 10, 10), Qt::black, 100.0f);
	DefaultTransferFunction.AddNode(0.7f, 1.0f, Qt::gray, QColor(10, 10, 10), Qt::black, 100.0f);
	DefaultTransferFunction.AddNode(1.0f, 1.0f, Qt::gray, QColor(10, 10, 10), Qt::black, 100.0f);

	return DefaultTransferFunction;
}