
#include "TransferFunction.h"

QNode::QNode(QTransferFunction* pTransferFunction, const float& Position, const float& Opacity, const QColor& Color, const bool& Deletable) :
	QObject(),
	m_pTransferFunction(pTransferFunction),
	m_Position(Position),
	m_Opacity(Opacity),
	m_Color(Color),
	m_Deletable(Deletable),
	m_AllowMoveH(Deletable),
	m_AllowMoveV(true),
	m_MinX(0.0f),
	m_MaxX(255.0f),
	m_MinY(0.0f),
	m_MaxY(1.0f)
{
}

float QNode::GetNormalizedX(void) const 
{
	return (GetPosition() - m_pTransferFunction->m_RangeMin) / m_pTransferFunction->m_Range;
}

void QNode::SetNormalizedX(const float& NormalizedX)
{
	SetPosition(m_pTransferFunction->m_RangeMin + (m_pTransferFunction->m_Range * NormalizedX));
}

float QNode::GetNormalizedY(void) const 
{
	return GetOpacity();
}

void QNode::SetNormalizedY(const float& NormalizedY)
{
	SetOpacity(NormalizedY);
}

float	QNode::GetPosition(void) const 						{	return m_Position; 								}				
void	QNode::SetPosition(const float& Position)				{	m_Position = Position; emit NodeChanged(this);}		
float	QNode::GetOpacity(void) const							{	return m_Opacity;								}
void	QNode::SetOpacity(const float& Opacity)				{	m_Opacity = Opacity; emit NodeChanged(this);	}
QColor	QNode::GetColor(void) const							{	return m_Color;									}
void	QNode::SetColor(const float& Color)					{	m_Color = Color; emit NodeChanged(this);		}
bool	QNode::GetAllowMoveH(void) const						{	return m_AllowMoveH;							}
void	QNode::SetAllowMoveH(const bool& AllowMoveH)			{	m_AllowMoveH = AllowMoveH;						}
bool	QNode::GetAllowMoveV(void) const						{	return m_AllowMoveV;							}
void	QNode::SetAllowMoveV(const bool& AllowMoveV)			{	m_AllowMoveV = AllowMoveV;						}
float	QNode::GetMinX(void) const								{	return m_MinX;									}
void	QNode::SetMinX(const float& MinX)						{	m_MinX = MinX;									}
float	QNode::GetMaxX(void) const								{	return m_MaxX;									}
void	QNode::SetMaxX(const float& MaxX)						{	m_MaxX = MaxX;									}

QTransferFunction::QTransferFunction(QObject* pParent) :
	QObject(pParent),
	m_Nodes(),
	m_RangeMin(0.0f),
	m_RangeMax(255.0f),
	m_Range(m_RangeMax - m_RangeMin),
	m_pSelectedNode(NULL),
	m_LinearGradient()
{
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

void QTransferFunction::OnNodeChanged(QNode* pNode)
{
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
	return m_Nodes.indexOf(pNode);
}