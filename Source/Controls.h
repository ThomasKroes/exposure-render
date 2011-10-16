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

class QColorPushButton : public QPushButton
{
	Q_OBJECT

public:
	QColorPushButton(QWidget* pParent = NULL);

	virtual QSize sizeHint() const;
	virtual void paintEvent(QPaintEvent* pPaintEvent);
	virtual void mousePressEvent(QMouseEvent* pEvent);

	int		GetMargin(void) const;
	void	SetMargin(const int& Margin);
	int		GetRadius(void) const;
	void	SetRadius(const int& Radius);
	QColor	GetColor(void) const;
	void	SetColor(const QColor& Color, bool BlockSignals = false);

private slots:
	void OnCurrentColorChanged(const QColor& Color);

signals:
	void currentColorChanged(const QColor&);

private:
	int		m_Margin;
	int		m_Radius;
	QColor	m_Color;
};

class QColorSelector : public QFrame
{
	Q_OBJECT

public:
	QColorSelector(QWidget* pParent = NULL);

//	virtual QSize sizeHint() const;

	QColor	GetColor(void) const;
	void	SetColor(const QColor& Color, bool BlockSignals = false);

private slots:
	void	OnCurrentColorChanged(const QColor& Color);

signals:
	void currentColorChanged(const QColor&);

private:
	QGridLayout			m_MainLayout;
	QColorPushButton	m_ColorButton;
	QComboBox			m_ColorCombo;
};

class QDoubleSlider : public QSlider
{
    Q_OBJECT

public:
    QDoubleSlider(QWidget* pParent = NULL);
	
	void setRange(double Min, double Max);
	void setMinimum(double Min);
	double minimum() const;
	void setMaximum(double Max);
	double maximum() const;
	double value() const;

public slots:
	void setValue(int value);
	void setValue(double Value, bool BlockSignals = false);

private slots:

signals:
	void valueChanged(double Value);
	void rangeChanged(double Min, double Max);

private:
	double	m_Multiplier;
};

class QDoubleSpinner : public QDoubleSpinBox
{
	Q_OBJECT

public:

	QDoubleSpinner(QWidget* pParent = NULL);;

	virtual QSize sizeHint() const;
	void setValue(double Value, bool BlockSignals = false);
};

class QInputDialogEx : public QInputDialog
{
public:
	QInputDialogEx(QWidget* pParent = NULL, Qt::WindowFlags Flags = 0);

	virtual QSize sizeHint() const;
};