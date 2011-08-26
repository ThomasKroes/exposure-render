#pragma once

#include <QtGui>

class QTransferFunction;

class QNode : public QObject
{
	Q_OBJECT

public:
	QNode::QNode(const QNode& Other)
	{
		*this = Other;
	};

	QNode(QTransferFunction* pTransferFunction, const float& Position, const float& Opacity, const QColor& Color, const bool& Deletable = true);

public:
	// From 2D
	float	GetX(void) const								{	return GetPosition();					}
	void	SetX(const float& X)							{	SetPosition(X);							}
	float	GetY(void) const								{	return GetOpacity();					}
	void	SetY(const float& Y)							{	SetOpacity(Y);							}
	
	float	GetNormalizedX(void) const;
	void	SetNormalizedX(const float& NormalizedX);


	float	GetPosition(void) const 						{	return m_Position; 								}				
	void	SetPosition(const float& Position)				{	m_Position = Position; emit NodeChanged(this);	}		
	float	GetOpacity(void) const							{	return m_Opacity;								}
	void	SetOpacity(const float& Opacity)				{	m_Opacity = Opacity; emit NodeChanged(this);	}
	QColor	GetColor(void) const							{	return m_Color;									}
	void	SetColor(const float& Color)					{	m_Color = Color; emit NodeChanged(this);		}
	bool	GetAllowMoveH(void) const						{	return m_AllowMoveH;							}
	void	SetAllowMoveH(const bool& AllowMoveH)			{	m_AllowMoveH = AllowMoveH;						}
	bool	GetAllowMoveV(void) const						{	return m_AllowMoveV;							}
	void	SetAllowMoveV(const bool& AllowMoveV)			{	m_AllowMoveV = AllowMoveV;						}
	float	GetMinX(void) const								{	return m_MinX;									}
	void	SetMinX(const float& MinX)						{	m_MinX = MinX;									}
	float	GetMaxX(void) const								{	return m_MaxX;									}
	void	SetMaxX(const float& MaxX)						{	m_MaxX = MaxX;									}

	QNode& operator = (const QNode& Other)			
	{
		m_pTransferFunction	= Other.m_pTransferFunction;
		m_Position			= Other.m_Position;
		m_Opacity			= Other.m_Opacity;
		m_Color				= Other.m_Color;
		m_Deletable			= Other.m_Deletable;
		m_AllowMoveH		= Other.m_AllowMoveH;
		m_AllowMoveV		= Other.m_AllowMoveV;

		return *this;
	}

signals:
	void NodeChanged(QNode* pNode);

public:
	QTransferFunction*		m_pTransferFunction;
	float					m_Position;
	float					m_Opacity;
	QColor					m_Color;
	bool					m_Deletable;
	bool					m_AllowMoveH;
	bool					m_AllowMoveV;
	float					m_MinX;
	float					m_MaxX;
};

class QTransferFunction : public QObject
{
    Q_OBJECT

public:
    QTransferFunction(QObject* pParent = NULL);

	void AddNode(QNode* pNode)
	{
		m_Nodes.append(pNode);

		// Emit
		emit NodeAdd(m_Nodes.back());
		emit FunctionChanged();

		connect(pNode, SIGNAL(NodeChanged(QNode*)), this, SLOT(OnNodeChanged(QNode*)));
	}

	void	SetSelectedNode(QNode* pSelectedNode);
	void	SetSelectedNode(const int& Index);
	void	SelectPreviousNode(void);
	void	SelectNextNode(void);
	int		GetNodeIndex(QNode* pNode);

private slots:
	void	OnNodeChanged(QNode* pNode);

signals:
	void	FunctionChanged(void);
	void	NodeAdd(QNode* pNode);
	void	NodeRemove(QNode* pNode);
	void	SelectionChanged(QNode* pNode);

public:
	QList<QNode*>		m_Nodes;
	float				m_RangeMin;
	float				m_RangeMax;
	float				m_Range;
	QNode*				m_pSelectedNode;
	QLinearGradient		m_LinearGradient;
};