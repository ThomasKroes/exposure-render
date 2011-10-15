#pragma once

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
	
	void						AddNode(const float& Intensity, const float& Opacity, const QColor& Diffuse, const QColor& Specular, const QColor& Emission, const float& Roughness);
	void						AddNode(const QNode& pNode);
	void						RemoveNode(QNode* pNode);
	void						UpdateNodeRanges(void);
	const QNodeList&			GetNodes(void) const;
	QNode&						GetNode(const int& Index);
	void						SetSelectedNode(QNode* pSelectedNode);
	void						SetSelectedNode(const int& Index);
	QNode*						GetSelectedNode(void);
	void						SelectFirstNode(void);
	void						SelectPreviousNode(void);
	void						SelectNextNode(void);
	void						SelectLastNode(void);
	int							GetNodeIndex(QNode* pNode);
	float						GetDensityScale(void) const;
	void						SetDensityScale(const float& DensityScale);
	int							GetShadingType(void) const;
	void						SetShadingType(const int& ShadingType);
	float						GetGradientFactor(void) const;
	void						SetGradientFactor(const float& GradientFactor);
	void						ReadXML(QDomElement& Parent);
	QDomElement					WriteXML(QDomDocument& DOM, QDomElement& Parent);
	static QTransferFunction	Default(void);

private slots:
	void	OnNodeChanged(QNode* pNode);

signals:
	void	Changed(void);
	void	SelectionChanged(QNode* pNode);
	
private:
	QNodeList	m_Nodes;
	QNode*		m_pSelectedNode;
	float		m_DensityScale;
	int			m_ShadingType;
	float		m_GradientFactor;

	friend class QNode;
};

typedef QList<QTransferFunction> QTransferFunctionList;

// Transfer function singleton
extern QTransferFunction gTransferFunction;