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

class QNode;

class QNodePropertiesWidget : public QGroupBox
{
    Q_OBJECT

public:
	QNodePropertiesWidget(QWidget* pParent = NULL);

private slots:
	void OnNodeSelectionChanged(QNode* pNode);
	void OnIntensityChanged(const double& Position);
	void OnOpacityChanged(const double& Opacity);
	void OnDiffuseChanged(const QColor& Diffuse);
	void OnSpecularChanged(const QColor& Specular);
	void OnEmissionChanged(const QColor& Emission);
	void OnGlossinessChanged(const double& Roughness);
	void OnNodeIntensityChanged(QNode* pNode);
	void OnNodeOpacityChanged(QNode* pNode);
	void OnNodeDiffuseChanged(QNode* pNode);
	void OnNodeSpecularChanged(QNode* pNode);
	void OnNodeEmissionChanged(QNode* pNode);
	void OnNodeGlossinessChanged(QNode* pNode);

private:
	QGridLayout		m_MainLayout;
	QLabel			m_IntensityLabel;
	QDoubleSlider	m_IntensitySlider;
	QDoubleSpinner	m_IntensitySpinBox;
	QLabel			m_OpacityLabel;
	QDoubleSlider	m_OpacitySlider;
	QDoubleSpinner	m_OpacitySpinBox;
	QColorSelector	m_Diffuse;
	QColorSelector	m_Specular;
	QLabel			m_GlossinessLabel;
	QDoubleSlider	m_GlossinessSlider;
	QDoubleSpinner	m_GlossinessSpinner;
	QColorSelector	m_Emission;
};