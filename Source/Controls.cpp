/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

QColorPushButton::QColorPushButton(QWidget* pParent) :
	QPushButton(pParent),
	m_Margin(5),
	m_Radius(4),
	m_Color(Qt::gray)
{
	setText("");
}

QSize QColorPushButton::sizeHint() const
{
	return QSize(20, 20);
}

void QColorPushButton::paintEvent(QPaintEvent* pPaintEvent)
{
	setText("");

	QPushButton::paintEvent(pPaintEvent);

	QPainter Painter(this);

	// Get button rectangle
	QRect ColorRectangle = pPaintEvent->rect();

	// Deflate it
	ColorRectangle.adjust(m_Margin, m_Margin, -m_Margin, -m_Margin);

	// Use anti aliasing
	Painter.setRenderHint(QPainter::Antialiasing);

	// Rectangle styling
	Painter.setBrush(QBrush(isEnabled() ? m_Color : Qt::lightGray));
	Painter.setPen(QPen(isEnabled() ? QColor(25, 25, 25) : Qt::darkGray, 0.5));

	// Draw
	Painter.drawRoundedRect(ColorRectangle, m_Radius, Qt::AbsoluteSize);
}

void QColorPushButton::mousePressEvent(QMouseEvent* pEvent)
{
	QColorDialog ColorDialog;

	connect(&ColorDialog, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));

	ColorDialog.setWindowIcon(GetIcon("color--pencil"));
	ColorDialog.setCurrentColor(m_Color);
	ColorDialog.exec();

	disconnect(&ColorDialog, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));
}

int QColorPushButton::GetMargin(void) const
{
	return m_Margin;
}

void QColorPushButton::SetMargin(const int& Margin)
{
	m_Margin = m_Margin;
	update();
}

int QColorPushButton::GetRadius(void) const
{
	return m_Radius;
}

void QColorPushButton::SetRadius(const int& Radius)
{
	m_Radius = m_Radius;
	update();
}

QColor QColorPushButton::GetColor(void) const
{
	return m_Color;
}

void QColorPushButton::SetColor(const QColor& Color, bool BlockSignals)
{
	blockSignals(BlockSignals);

	m_Color = Color;
	update();

	blockSignals(false);
}

void QColorPushButton::OnCurrentColorChanged(const QColor& Color)
{
	SetColor(Color);

	emit currentColorChanged(m_Color);
}

QColorSelector::QColorSelector(QWidget* pParent /*= NULL*/) :
	QFrame(pParent),
	m_ColorButton(),
	m_ColorCombo()
{
	setLayout(&m_MainLayout);

	m_MainLayout.addWidget(&m_ColorButton, 0, 0, Qt::AlignLeft);
//	m_MainLayout.addWidget(&m_ColorCombo, 0, 1);

	m_MainLayout.setContentsMargins(0, 0, 0, 0);
	
	m_ColorButton.setFixedWidth(30);

	QObject::connect(&m_ColorButton, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));
}

QColor QColorSelector::GetColor(void) const
{
	return m_ColorButton.GetColor();
}

void QColorSelector::SetColor(const QColor& Color, bool BlockSignals /*= false*/)
{
	m_ColorButton.SetColor(Color, BlockSignals);
}

void QColorSelector::OnCurrentColorChanged(const QColor& Color)
{
	emit currentColorChanged(Color);
}

QDoubleSlider::QDoubleSlider(QWidget* pParent /*= NULL*/) :
	QSlider(pParent),
	m_Multiplier(10000.0)
{
	connect(this, SIGNAL(valueChanged(int)), this, SLOT(setValue(int)));

	setSingleStep(1);

	setOrientation(Qt::Horizontal);
	setFocusPolicy(Qt::NoFocus);
}

void QDoubleSlider::setValue(int Value)
{
	emit valueChanged((double)Value / m_Multiplier);
}

void QDoubleSlider::setValue(double Value, bool BlockSignals)
{
	QSlider::blockSignals(BlockSignals);

	QSlider::setValue(Value * m_Multiplier);

 	if (!BlockSignals)
 		emit valueChanged(Value);

	QSlider::blockSignals(false);
}

void QDoubleSlider::setRange(double Min, double Max)
{
	QSlider::setRange(Min * m_Multiplier, Max * m_Multiplier);

	emit rangeChanged(Min, Max);
}

void QDoubleSlider::setMinimum(double Min)
{
	QSlider::setMinimum(Min * m_Multiplier);

	emit rangeChanged(minimum(), maximum());
}

double QDoubleSlider::minimum() const
{
	return QSlider::minimum() / m_Multiplier;
}

void QDoubleSlider::setMaximum(double Max)
{
	QSlider::setMaximum(Max * m_Multiplier);

	emit rangeChanged(minimum(), maximum());
}

double QDoubleSlider::maximum() const
{
	return QSlider::maximum() / m_Multiplier;
}

double QDoubleSlider::value() const
{
	int Value = QSlider::value();
	return (double)Value / m_Multiplier;
}

QSize QDoubleSpinner::sizeHint() const
{
	return QSize(90, 20);
}

QDoubleSpinner::QDoubleSpinner(QWidget* pParent /*= NULL*/) :
	QDoubleSpinBox(pParent)
{
}

void QDoubleSpinner::setValue(double Value, bool BlockSignals)
{
	blockSignals(BlockSignals);

	QDoubleSpinBox::setValue(Value);

	blockSignals(false);
}

QInputDialogEx::QInputDialogEx(QWidget* pParent /*= NULL*/, Qt::WindowFlags Flags /*= 0*/) :
	QInputDialog(pParent, Flags)
{
	setWindowIcon(GetIcon("pencil-field"));
}

QSize QInputDialogEx::sizeHint() const
{
	return QSize(350, 60);
}