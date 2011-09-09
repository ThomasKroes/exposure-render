#pragma once

#include <QtGui>

#include "Preset.h"
#include "Node.h"
#include "Histogram.h"

class QTransferFunction : public QPresetXML
{
    Q_OBJECT

public:
    QTransferFunction(QObject* pParent = NULL, const QString& Name = "Default");
	QTransferFunction(const QTransferFunction& Other);
	QTransferFunction& operator = (const QTransferFunction& Other);			
	
	void				AddNode(const float& Intensity, const float& Opacity, const QColor& Diffuse, const QColor& SpecularColor, const float& Roughness);
	void				AddNode(const QNode& pNode);
	void				RemoveNode(QNode* pNode);
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
	void				ReadXML(QDomElement& Parent);
	QDomElement			WriteXML(QDomDocument& DOM, QDomElement& Parent);

	static QTransferFunction	Default(void);
	static float				GetRangeMin(void);
	static void					SetRangeMin(const float& RangeMin);
	static float				GetRangeMax(void);
	static void					SetRangeMax(const float& RangeMax);
	static float				GetRange(void);

private slots:
	void	OnNodeChanged(QNode* pNode);

signals:
	void	FunctionChanged(void);
	void	SelectionChanged(QNode* pNode);
	
protected:
	QNodeList		m_Nodes;
	QNode*			m_pSelectedNode;
	static float	m_RangeMin;
	static float	m_RangeMax;
	static float	m_Range;

	friend class QNode;
};

typedef QList<QTransferFunction> QTransferFunctionList;

// Transfer function singleton
extern QTransferFunction gTransferFunction;