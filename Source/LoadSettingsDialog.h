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

class QLoadSettingsDialog : public QDialog
{
	Q_OBJECT

public:
    QLoadSettingsDialog(QWidget* pParent = NULL);

	virtual void accept();

public:
	bool GetResample(void);
	float GetResampleX(void);
	float GetResampleY(void);
	float GetResampleZ(void);

	void Reset(void);
	void SetToolTips(void);

private slots:
	void LockY(const int& State);
	void LockZ(const int& State);
	void SetResample(const int& Resample);
	void SetResampleX(const double& ResampleX);
	void SetResampleY(const double& ResampleY);
	void SetResampleZ(const double& ResampleZ);
	void Accept(void);
	void Reject(void);
	void Clicked(QAbstractButton* pButton);

private:
	QGridLayout			m_MainLayout;
	QGroupBox			m_ResampleGroupBox;
	QGridLayout			m_ResampleLayout;
	QLabel				m_ResampleXLabel;
	QDoubleSlider		m_ResampleXSlider;
	QDoubleSpinner		m_ResampleXSpinBox;
	QLabel				m_ResampleYLabel;
	QDoubleSlider		m_ResampleYSlider;
	QDoubleSpinner		m_ResampleYSpinBox;
	QCheckBox			m_LockYCheckBox;
	QLabel				m_ResampleZLabel;
	QDoubleSlider		m_ResampleZSlider;
	QDoubleSpinner		m_ResampleZSpinBox;
	QCheckBox			m_LockZCheckBox;
	QDialogButtonBox	m_DialogButtons;

	bool				m_Resample;
	float				m_ResampleX;
	float				m_ResampleY;
	float				m_ResampleZ;
};