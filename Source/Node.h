/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Preset.h"

class QTransferFunction;

class QNode : public QPresetXML
{
	Q_OBJECT

public:
	QNode(const QNode& Other);
	QNode(QTransferFunction* pTransferFunction, const float& Intensity = 0.0f, const float& Opacity = 0.5f, const QColor& Diffuse = Qt::white, const QColor& Specular = Qt::white, const QColor& Emission = Qt::black, const float& Roughness = 100.0f);
	QNode& operator = (const QNode& Other);
	bool operator == (const QNode& Other) const;

	QTransferFunction*	GetTransferFunction(void);
	float				GetIntensity(void) const;
	void				SetIntensity(const float& Intensity);
	float				GetOpacity(void) const;
	void				SetOpacity(const float& Opacity);
	QColor				GetDiffuse(void) const;
	void				SetDiffuse(const QColor& Diffuse);
	QColor				GetSpecular(void) const;
	void				SetSpecular(const QColor& Specular); 
	QColor				GetEmission(void) const;
	void				SetEmission(const QColor& Emission);
	float				GetGlossiness(void) const;
	void				SetGlossiness(const float& Roughness);
	float				GetMinX(void) const;
	void				SetMinX(const float& MinX);
	float				GetMaxX(void) const;
	void				SetMaxX(const float& MaxX);
	float				GetMinY(void) const;
	void				SetMinY(const float& MinY);
	float				GetMaxY(void) const;
	void				SetMaxY(const float& MaxY);
	bool				InRange(const QPointF& Point);
	int					GetID(void) const;
	bool				GetDirty(void) const;
	void				SetDirty(const bool& Dirty);
	void				ReadXML(QDomElement& Parent);
	QDomElement			WriteXML(QDomDocument& DOM, QDomElement& Parent);

signals:
	void NodeChanged(QNode* pNode);
	void IntensityChanged(QNode* pNode);
	void OpacityChanged(QNode* pNode);
	void DiffuseChanged(QNode* pNode);
	void SpecularChanged(QNode* pNode);
	void EmissionChanged(QNode* pNode);
	void RoughnessChanged(QNode* pNode);
	void RangeChanged(QNode* pNode);

protected:
	QTransferFunction*	m_pTransferFunction;
	float				m_Intensity;
	float				m_Opacity;
	QColor				m_Diffuse;
	QColor				m_Specular;
	QColor				m_Emission;
	float				m_Glossiness;
	float				m_MinX;
	float				m_MaxX;
	float				m_MinY;
	float				m_MaxY;
	int					m_ID;
	bool				m_Dirty;

	friend class QTransferFunction;
};

typedef QList<QNode> QNodeList;