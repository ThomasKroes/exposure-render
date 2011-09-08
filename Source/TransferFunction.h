#pragma once

#include <QtGui>

#include "Preset.h"

class QTransferFunction;

class QHistogram : public QObject
{
	Q_OBJECT

public:
    QHistogram(QObject* pParent = NULL);
	QHistogram::QHistogram(const QHistogram& Other);
	QHistogram& operator = (const QHistogram& Other);

	bool			GetEnabled(void) const;
	void			SetEnabled(const bool& Enabled);
	QList<int>&		GetBins(void);
	void			SetBins(const QList<int>& Bins);
	int				GetMax(void) const;
	void			SetMax(const int& Max);
	void			Reset(void);

signals:
	void HistogramChanged(void);

private:
	bool			m_Enabled;
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

	QNode(QTransferFunction* pTransferFunction, const float& Intensity = 0.0f, const float& Opacity = 0.5f, const QColor& Diffuse = Qt::white, const QColor& Specular = Qt::white, const QColor& Emission = Qt::black, const float& Roughness = 100.0f);
	
	QNode& operator = (const QNode& Other);

	bool operator == (const QNode& Other) const;

	float		GetIntensity(void) const;
	void		SetIntensity(const float& Intensity);
	float		GetNormalizedIntensity(void) const;
	void		SetNormalizedIntensity(const float& NormalizedX);
	float		GetOpacity(void) const;
	void		SetOpacity(const float& Opacity);
	QColor		GetDiffuse(void) const;
	void		SetDiffuse(const QColor& Diffuse);
	QColor		GetSpecular(void) const;
	void		SetSpecular(const QColor& Specular);
	QColor		GetEmission(void) const;
	void		SetEmission(const QColor& Emission);
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
	QColor				m_Diffuse;
	QColor				m_Specular;
	QColor				m_Emission;
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
	
	void				AddNode(const float& Intensity, const float& Opacity, const QColor& DiffuseColor, const QColor& SpecularColor, const float& Roughness);
	void				AddNode(const QNode& pNode);
	void				RemoveNode(QNode* pNode);
	float				GetRangeMin(void) const;
	void				SetRangeMin(const float& RangeMin);
	float				GetRangeMax(void) const;
	void				SetRangeMax(const float& RangeMax);
	float				GetRange(void) const;
	void				NormalizeIntensity(void);
	void				DeNormalizeIntensity(void);
	void				UpdateNodeRanges(void);
	const QNodeList&	GetNodes(void) const;
	QNode&				GetNode(const int& Index);
	void				SetSelectedNode(QNode* pSelectedNode);
	void				SetSelectedNode(const int& Index);
	QNode*				GetSelectedNode(void);
	void				SelectPreviousNode(void);
	void				SelectNextNode(void);
	int					GetNodeIndex(QNode* pNode);
	QHistogram&			GetHistogram(void);		
	void				SetHistogram(const int* pBins, const int& NoBins);
	void				ReadXML(QDomElement& Parent);
	QDomElement			WriteXML(QDomDocument& DOM, QDomElement& Parent);

	static QTransferFunction Default(void);

private slots:
	void	OnNodeChanged(QNode* pNode);

signals:
	void	FunctionChanged(void);
	void	SelectionChanged(QNode* pNode);
	
protected:
	QNodeList		m_Nodes;
	QNode*			m_pSelectedNode;
	QHistogram		m_Histogram;

	static float	m_RangeMin;
	static float	m_RangeMax;
	static float	m_Range;

	friend class QNode;
};

typedef QList<QTransferFunction> QTransferFunctionList;

// Transfer function singleton
extern QTransferFunction gTransferFunction;