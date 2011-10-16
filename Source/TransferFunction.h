/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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