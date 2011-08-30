#pragma once

#include <QtGui>

#include <QtXml\qdom.h>

class QTransferFunction;

class QHistogram : public QGraphicsPolygonItem
{
public:
    QHistogram(QGraphicsItem* pParent = NULL) :
		QGraphicsPolygonItem(pParent)
	{
	}

	QHistogram::QHistogram(const QHistogram& Other)
	{
		*this = Other;
	};

	QHistogram& operator = (const QHistogram& Other)			
	{
		m_Bins	= Other.m_Bins;
		m_Max	= Other.m_Max;

		return *this;
	}

	QList<int>		m_Bins;
	int				m_Max;
};

class QNode : public QObject
{
	Q_OBJECT

public:
	QNode::QNode(const QNode& Other)
	{
		*this = Other;
	};

	QNode(QTransferFunction* pTransferFunction, const float& Position = 0.0f, const float& Opacity = 0.0f, const QColor& Color = Qt::black);
	
	QNode& operator = (const QNode& Other)			
	{
		m_Position	= Other.m_Position;
		m_Opacity	= Other.m_Opacity;
		m_Color		= Other.m_Color;
		m_MinX		= Other.m_MinX;
		m_MaxX		= Other.m_MaxX;
		m_MinY		= Other.m_MinY;
		m_MaxY		= Other.m_MaxY;

		return *this;
	}

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
	float	GetMinX(void) const;
	void	SetMinX(const float& MinX);
	float	GetMaxX(void) const;
	void	SetMaxX(const float& MaxX);
	float	GetMinY(void) const;
	void	SetMinY(const float& MinY);
	float	GetMaxY(void) const;
	void	SetMaxY(const float& MaxY);
	bool	InRange(const QPointF& Point);
	void	ReadXML(QDomElement& Parent);
	void	WriteXML(QDomDocument& DOM, QDomElement& Parent);

signals:
	void NodeChanged(QNode* pNode);
	void PositionChanged(QNode* pNode);
	void OpacityChanged(QNode* pNode);
	void ColorChanged(QNode* pNode);
	void RangeChanged(QNode* pNode);

public:
	QTransferFunction*	m_pTransferFunction;
	float				m_Position;
	float				m_Opacity;
	QColor				m_Color;
	float				m_MinX;
	float				m_MaxX;
	float				m_MinY;
	float				m_MaxY;
};

class QTransferFunction : public QObject
{
    Q_OBJECT

public:
    QTransferFunction(QObject* pParent = NULL, const QString& Name = "Default");

	QTransferFunction::QTransferFunction(const QTransferFunction& Other)
	{
		*this = Other;
	};

	QTransferFunction& operator = (const QTransferFunction& Other)			
	{
		m_Name				= Other.m_Name;
		m_Nodes				= Other.m_Nodes;
		m_RangeMin			= Other.m_RangeMin;
		m_RangeMax			= Other.m_RangeMax;
		m_Range				= Other.m_Range;
		m_pSelectedNode		= Other.m_pSelectedNode;
		m_Histogram			= Other.m_Histogram;

		return *this;
	}

	QString	GetName(void) const;
	void	SetName(const QString& Name);
	void	AddNode(const float& Position, const float& Opacity, const QColor& Color);
	void	AddNode(QNode* pNode);
	void	RemoveNode(QNode* pNode);
	void	SetSelectedNode(QNode* pSelectedNode);
	void	SetSelectedNode(const int& Index);
	void	SelectPreviousNode(void);
	void	SelectNextNode(void);
	int		GetNodeIndex(QNode* pNode);
	void	UpdateNodeRanges(void);
	void	SetHistogram(const int* pBins, const int& NoBins);
	void	ReadXML(QDomElement& Parent);
	void	WriteXML(QDomDocument& DOM, QDomElement& Parent);

private slots:
	void	OnNodeChanged(QNode* pNode);

signals:
	void	FunctionChanged(void);
	void	NodeAdd(QNode* pNode);
	void	NodeRemove(QNode* pNode);
	void	NodeRemoved(QNode* pNode);
	void	SelectionChanged(QNode* pNode);
	void	HistogramChanged(void);

public:
	QString				m_Name;
	QVector<QNode*>		m_Nodes;
	float				m_RangeMin;
	float				m_RangeMax;
	float				m_Range;
	QNode*				m_pSelectedNode;
	QHistogram			m_Histogram;
};

Q_DECLARE_METATYPE(QTransferFunction)

// Transfer function singleton
extern QTransferFunction gTransferFunction;