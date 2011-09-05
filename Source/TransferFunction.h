#pragma once

#include <QtGui>

#include "Preset.h"

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

class QNode : public QPresetXML
{
	Q_OBJECT

public:
	QNode::QNode(const QNode& Other)
	{
		*this = Other;
	};

	QNode(QTransferFunction* pTransferFunction, const float& Intensity = 0.0f, const float& Opacity = 0.5f, const QColor& DiffuseColor = Qt::white, const QColor& SpecularColor = Qt::white, const float& Roughness = 100.0f);
	
	QNode& operator = (const QNode& Other)			
	{
		m_pTransferFunction	= Other.m_pTransferFunction;
		m_Intensity			= Other.m_Intensity;
		m_Opacity			= Other.m_Opacity;
		m_DiffuseColor		= Other.m_DiffuseColor;
		m_SpecularColor		= Other.m_SpecularColor;
		m_Roughness			= Other.m_Roughness;
		m_MinX				= Other.m_MinX;
		m_MaxX				= Other.m_MaxX;
		m_MinY				= Other.m_MinY;
		m_MaxY				= Other.m_MaxY;
		m_ID				= Other.m_ID;

		return *this;
	}

	bool operator == (const QNode& Other) const
	{
		return m_ID == Other.m_ID;
	}

	float		GetNormalizedIntensity(void) const;
	void		SetNormalizedIntensity(const float& NormalizedX);
	float		GetNormalizedOpacity(void) const;
	void		SetNormalizedOpacity(const float& NormalizedY);
	float		GetIntensity(void) const;
	void		SetIntensity(const float& Intensity);
	float		GetOpacity(void) const;
	void		SetOpacity(const float& Opacity);
	QColor		GetDiffuseColor(void) const;
	void		SetDiffuseColor(const QColor& DiffuseColor);
	QColor		GetSpecularColor(void) const;
	void		SetSpecularColor(const QColor& SpecularColor);
	float		GetRoughness(void) const;
	void		SetRoughness(const float& Roughness);
	float		GetMinX(void) const;
	void		SetMinX(const float& MinX);
	float		GetMaxX(void) const;
	void		SetMaxX(const float& MaxX);
	float		GetMinY(void) const;
	void		SetMinY(const float& MinY);
	float		GetMaxY(void) const;
	void		SetMaxY(const float& MaxY);
	bool		InRange(const QPointF& Point);
	int			GetID(void) const;
	void		ReadXML(QDomElement& Parent);
	QDomElement	WriteXML(QDomDocument& DOM, QDomElement& Parent);

signals:
	void NodeChanged(QNode* pNode);
	void IntensityChanged(QNode* pNode);
	void OpacityChanged(QNode* pNode);
	void DiffuseColorChanged(QNode* pNode);
	void SpecularColorChanged(QNode* pNode);
	void RoughnessChanged(QNode* pNode);
	void RangeChanged(QNode* pNode);

protected:
	QTransferFunction*	m_pTransferFunction;
	float				m_Intensity;
	float				m_Opacity;
	QColor				m_DiffuseColor;
	QColor				m_SpecularColor;
	float				m_Roughness;
	float				m_MinX;
	float				m_MaxX;
	float				m_MinY;
	float				m_MaxY;
	int					m_ID;

	friend class QTransferFunction;
};

typedef QList<QNode> QNodeList;

class QTransferFunction : public QPresetXML
{
    Q_OBJECT

public:
    QTransferFunction(QObject* pParent = NULL, const QString& Name = "Default");

	QTransferFunction::QTransferFunction(const QTransferFunction& Other)
	{
		*this = Other;
	};

	QTransferFunction& operator = (const QTransferFunction& Other);			
	
	void				AddNode(const float& Position, const float& Opacity, const QColor& DiffuseColor, const QColor& SpecularColor, const float& Roughness);
	void				AddNode(const QNode& pNode);
	void				RemoveNode(QNode* pNode);
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
	QDomElement			WriteXML(QDomDocument& DOM, QDomElement& Parent);

private slots:
	void	OnNodeChanged(QNode* pNode);

signals:
	void	FunctionChanged(void);
	void	SelectionChanged(QNode* pNode);
	void	NodeCountChange();
	void	NodeCountChanged();
	void	HistogramChanged(void);

protected:
	QNodeList		m_Nodes;
	float			m_RangeMin;
	float			m_RangeMax;
	float			m_Range;
	QNode*			m_pSelectedNode;
	QHistogram		m_Histogram;

	friend class QNode;
};

typedef QList<QTransferFunction> QTransferFunctionList;

// Transfer function singleton
extern QTransferFunction gTransferFunction;