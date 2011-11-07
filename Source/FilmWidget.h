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

#include "Film.h"

class QFilmResolutionPreset
{
public:
	QFilmResolutionPreset(void) :
		m_Width(0),
		m_Height(0)
	{
	}

	QFilmResolutionPreset(const int& Width, const int& Height) :
		m_Width(Width),
		m_Height(Height)
	{
	}

	int		GetWidth(void) const			{ return m_Width;		}
	void	SetWidth(const int& Width)		{ m_Width = Width;		}
	int		GetHeight(void) const			{ return m_Height;		}
	void	SetHeight(const int& Height)	{ m_Height = Height;	}

private:
	int	m_Width;
	int	m_Height;
};

class QFilmResolutionButton : public QPushButton
{
	Q_OBJECT

public:
	QFilmResolutionButton(void)
	{
	};

	QFilmResolutionButton(const int& Width, const int& Height) :
		m_Preset(Width, Height)
	{
	}

	void SetPreset(const int& Width, const int& Height)
	{
		m_Preset.SetWidth(Width);
		m_Preset.SetHeight(Height);

		const QString Message = QString::number(Width) + " x " + QString::number(Height);

		setText(Message);
		setToolTip(Message);
		setStatusTip("Change render resolution to " + Message);
	}

	void mousePressEvent(QMouseEvent* pEvent)
	{
		emit SetPreset(m_Preset);
	}

signals:
	void SetPreset(QFilmResolutionPreset& Preset);

private:
	QFilmResolutionPreset	m_Preset;
};

class QFilmWidget : public QGroupBox
{
    Q_OBJECT

public:
    QFilmWidget(QWidget* pParent = NULL);

public slots:
	void SetPresetType(const QString& PresetType);
	void SetPreset(QFilmResolutionPreset& Preset);
	void SetWidth(const int& Width);
	void SetHeight(const int& Height);
	void SetExposure(const double& Exposure);
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void OnFilmChanged(const QFilm& Film);
	void OnNoiseReduction(const int& ReduceNoise);

private:
	QGridLayout				m_GridLayout;
	QComboBox				m_PresetType;
	QGridLayout				m_PresetsLayout;
	QFilmResolutionButton	m_Preset[4];
	QSpinBox				m_WidthSpinner;
	QSpinBox				m_HeightSpinner;
	QDoubleSlider			m_ExposureSlider;
	QDoubleSpinner			m_ExposureSpinner;
	QCheckBox				m_NoiseReduction;
};