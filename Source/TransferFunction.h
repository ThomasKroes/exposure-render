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
		m_Intensity	= Other.m_Intensity;
		m_Opacity	= Other.m_Opacity;
		m_Color		= Other.m_Color;
		m_MinX		= Other.m_MinX;
		m_MaxX		= Other.m_MaxX;
		m_MinY		= Other.m_MinY;
		m_MaxY		= Other.m_MaxY;
		m_GUID		= Other.m_GUID;

		return *this;
	}

	bool operator == (const QNode& Other) const
	{
		return GetGUID() == Other.GetGUID();
	}

	float	GetNormalizedIntensity(void) const;
	void	SetNormalizedIntensity(const float& NormalizedX);
	float	GetNormalizedOpacity(void) const;
	void	SetNormalizedOpacity(const float& NormalizedY);
	float	GetIntensity(void) const;
	void	SetIntensity(const float& Position);
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
	QString	GetGUID(void) const;
	void	ReadXML(QDomElement& Parent);
	void	WriteXML(QDomDocument& DOM, QDomElement& Parent);

signals:
	void NodeChanged(QNode* pNode);
	void PositionChanged(QNode* pNode);
	void OpacityChanged(QNode* pNode);
	void ColorChanged(QNode* pNode);
	void RangeChanged(QNode* pNode);

private:
	QTransferFunction*	m_pTransferFunction;
	float				m_Intensity;
	float				m_Opacity;
	QColor				m_Color;
	float				m_MinX;
	float				m_MaxX;
	float				m_MinY;
	float				m_MaxY;
	QUuid				m_GUID;
};

typedef QList<QNode> QNodeList;

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
		blockSignals(true);

		m_Name			= Other.m_Name;
		m_Nodes			= Other.m_Nodes;
		m_RangeMin		= Other.m_RangeMin;
		m_RangeMax		= Other.m_RangeMax;
		m_Range			= Other.m_Range;
		m_pSelectedNode	= Other.m_pSelectedNode;
		m_Histogram		= Other.m_Histogram;

		blockSignals(true);

		// Notify others that the function has changed selection has changed
		emit FunctionChanged();

		// Notify others that our selection has changed
		emit SelectionChanged(m_pSelectedNode);

		return *this;
	}

	void				AddNode(const float& Position, const float& Opacity, const QColor& Color);
	void				AddNode(const QNode& pNode);
	void				RemoveNode(QNode* pNode);
	QString				GetName(void) const;
	void				SetName(const QString& Name);
	float				GetRangeMin(void) const;
	void				SetRangeMin(const float& RangeMin);
	float				GetRangeMax(void) const;
	void				SetRangeMax(const float& RangeMax);
	float				GetRange(void) const;
	void				UpdateNodeRanges(void);
	const QNodeList&	GetNodes(void) const;
	QNode&				GetNode(const int& Index);
	void				SetSelectedNode(QNode* pSelectedNode);
	void				SetSelectedNode(const int& Index);
	QNode*				GetSelectedNode(void);
	void				SelectPreviousNode(void);
	void				SelectNextNode(void);
	int					GetNodeIndex(QNode* pNode);
	const QHistogram&	GetHistogram(void) const;		
	void				SetHistogram(const int* pBins, const int& NoBins);
	void				ReadXML(QDomElement& Parent);
	void				WriteXML(QDomDocument& DOM, QDomElement& Parent);

private slots:
	void	OnNodeChanged(QNode* pNode);

signals:
	void	FunctionChanged(void);
	void	NodeAdd(QNode* pNode);
	void	NodeRemove(QNode* pNode);
	void	NodeRemoved(QNode* pNode);
	void	SelectionChanged(QNode* pNode);
	void	HistogramChanged(void);

protected:
	QString		m_Name;
	QNodeList	m_Nodes;
	float		m_RangeMin;
	float		m_RangeMax;
	float		m_Range;
	QNode*		m_pSelectedNode;
	QHistogram	m_Histogram;

	friend class QNode;
};

Q_DECLARE_METATYPE(QTransferFunction)

// Transfer function singleton
extern QTransferFunction gTransferFunction;