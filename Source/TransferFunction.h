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

	QNode(QTransferFunction* gTransferFunction, const float& Position, const float& Opacity, const QColor& Color, const bool& Deletable = true);

	bool operator < (const QNode& Other) const
	{
		return GetPosition() > Other.GetPosition();
    }

public:
	// From 2D
	float	GetX(void) const								{	return GetPosition();					}
	void	SetX(const float& X)							{	SetPosition(X);							}
	float	GetY(void) const								{	return GetOpacity();					}
	void	SetY(const float& Y)							{	SetOpacity(Y);							}
	
	float	GetNormalizedX(void) const;
	void	SetNormalizedX(const float& NormalizedX);
	float	GetNormalizedY(void) const;
	void	SetNormalizedY(const float& NormalizedY);


	float	GetPosition(void) const;
	void	SetPosition(const float& Position);
	float	GetOpacity(void) const;
	void	SetOpacity(const float& Opacity);
	QColor	GetColor(void) const;
	void	SetColor(const QColor& Color);
	bool	GetAllowMoveH(void) const;
	void	SetAllowMoveH(const bool& AllowMoveH);
	bool	GetAllowMoveV(void) const;
	void	SetAllowMoveV(const bool& AllowMoveV);
	float	GetMinX(void) const;
	void	SetMinX(const float& MinX);
	float	GetMaxX(void) const;
	void	SetMaxX(const float& MaxX);
	float	GetMinY(void) const;
	void	SetMinY(const float& MinY);
	float	GetMaxY(void) const;
	void	SetMaxY(const float& MaxY);

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
	void PositionChanged(QNode* pNode);
	void OpacityChanged(QNode* pNode);
	void ColorChanged(QNode* pNode);
	void RangeChanged(QNode* pNode);

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
	float					m_MinY;
	float					m_MaxY;
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

// Transfer function singleton
extern QTransferFunction gTransferFunction;