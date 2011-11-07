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

class QLight;

class QLightWidget : public QGroupBox
{
    Q_OBJECT

public:
    QLightWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const { return QSize(10, 10); }

public slots:
	void OnLightSelectionChanged(QLight* pLight);

	void OnThetaChanged(const double& Theta);
	void OnPhiChanged(const double& Phi);
	void OnDistanceChanged(const double& Distance);
	void OnWidthChanged(const double& Width);
	void OnLockSizeChanged(int LockSize);
	void OnHeightChanged(const double& Height);
	void OnCurrentColorChanged(const QColor& Color);
	void OnIntensityChanged(const double& Intensity);

	void OnLightThetaChanged(QLight* pLight);
	void OnLightPhiChanged(QLight* pLight);
	void OnLightDistanceChanged(QLight* pLight);
	void OnLightWidthChanged(QLight* pLight);
	void OnLightLockSizeChanged(QLight* pLight);
	void OnLightHeightChanged(QLight* pLight);
	void OnLightColorChanged(QLight* pLight);
	void OnLightIntensityChanged(QLight* pLight);

protected:
	QGridLayout			m_MainLayout;
	QLabel				m_ThetaLabel;
	QDoubleSlider		m_ThetaSlider;
	QDoubleSpinner		m_ThetaSpinBox;
	QLabel				m_PhiLabel;
	QDoubleSlider		m_PhiSlider;
	QDoubleSpinner		m_PhiSpinBox;
	QLabel				m_DistanceLabel;
	QDoubleSlider		m_DistanceSlider;
	QDoubleSpinner		m_DistanceSpinner;
	QLabel				m_WidthLabel;
	QDoubleSlider		m_WidthSlider;
	QDoubleSpinner		m_WidthSpinner;
	QLabel				m_HeightLabel;
	QDoubleSlider		m_HeightSlider;
	QDoubleSpinner		m_HeightSpinner;
	QCheckBox			m_LockSizeCheckBox;
	QLabel				m_ColorLabel;
	QColorPushButton	m_ColorButton;
	QLabel				m_IntensityLabel;
	QDoubleSlider		m_IntensitySlider;
	QDoubleSpinner		m_IntensitySpinBox;
};