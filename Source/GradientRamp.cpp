/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "GradientRamp.h"

QGradientRamp::QGradientRamp(const QString& Name, QWidget* pParent /*= NULL*/) :
	QWidget(pParent),
	m_Name(Name.toLower()),
	m_GradientStops(),
	m_LinearGradient(),
	m_CheckerDimEnabled(QColor::fromHsl(0, 0, 160)),
	m_CheckerDimDisabled(QColor::fromHsl(0, 0, 180)),
	m_CheckerBrightEnabled(QColor::fromHsl(0, 0, 230)),
	m_CheckerBrightDisabled(QColor::fromHsl(0, 0, 210)),
	m_PenEnabled(QColor::fromHsl(0, 0, 100)),
	m_PenDisabled(QColor::fromHsl(0, 0, 150)),
	m_Font("Arial"),
	m_TextForegroundEnabled(QColor::fromHsl(0, 0, 230)),
	m_TextForegroundDisabled(QColor::fromHsl(0, 0, 230)),
	m_TextBackgroundEnabled(QColor::fromHsl(0, 0, 80)),
	m_TextBackgroundDisabled(QColor::fromHsl(0, 0, 150))
{
	m_Font.setPointSize(7);
}

void QGradientRamp::paintEvent(QPaintEvent * pe)
{
	QPainter Painter(this);

	const float CheckerSize = 0.5f * rect().height();
	
	const int NumX = ceilf((float)rect().width() / CheckerSize);

	QColor CheckerDim		= isEnabled() ? m_CheckerDimEnabled : m_CheckerDimDisabled;
	QColor CheckerBright	= isEnabled() ? m_CheckerBrightEnabled : m_CheckerBrightDisabled;

	for (int i = 0; i < NumX; i++)
	{
		const float Width = (i == NumX - 1) ? rect().width() - (NumX - 1) * CheckerSize : CheckerSize;

		QRectF RectTop(i * CheckerSize, 0, Width, CheckerSize);
		QRectF RectBottom(i * CheckerSize, CheckerSize, Width, CheckerSize);

		QBrush BrushTop(i % 2 == 0 ? CheckerDim : CheckerBright);
		QBrush BrushBottom(i % 2 == 0 ? CheckerBright : CheckerDim);

		Painter.fillRect(RectTop, BrushTop);
		Painter.fillRect(RectBottom, BrushBottom);
	}

	QGradientStops GradientStops = m_GradientStops, TextGradientStops = GradientStops;

	if (!isEnabled())
	{
		for (int i = 0; i < GradientStops.size(); i++)
		{
			const QColor OriginalColor = GradientStops[i].second;

			GradientStops[i].second = QColor::fromHsl(OriginalColor.hue(), 0, 200, 0.8 * OriginalColor.alpha());
		}
	}

	for (int i = 0; i < GradientStops.size(); i++)
	{
		const QColor OriginalColor = GradientStops[i].second;
		GradientStops[i].second.setAlphaF(0.8 * GradientStops[i].second.alphaF());
	}

	for (int i = 0; i < TextGradientStops.size(); i++)
	{
		TextGradientStops[i].second = QColor::fromHsl(0, 0, 255 - TextGradientStops[i].second.lightness(), 255);
	}
	

	m_LinearGradient.setStops(GradientStops);

	Painter.setPen(isEnabled() ? QPen(Qt::darkGray) : QPen(Qt::lightGray));

	// Draw the gradient
	Painter.fillRect(rect(), QBrush(m_LinearGradient));

	QRect R(rect());

	R.adjust(0, 0, -1, -1);

	Painter.setPen(isEnabled() ? m_PenEnabled : m_PenDisabled);

	Painter.drawRect(R);

	Painter.setFont(m_Font);
	
	Painter.setPen(QPen(isEnabled() ? m_TextBackgroundEnabled : m_TextBackgroundDisabled));

	Painter.drawText(rect(), Qt::AlignCenter, m_Name);

	Painter.translate(-0.55, -0.25);

	Painter.setFont(m_Font);
	Painter.setPen(QPen(isEnabled() ? m_TextForegroundEnabled : m_TextForegroundDisabled));

	const QRectF RectWhite = QRectF(-0.55, -0.25, rect().width(), rect().height());

	Painter.drawText(RectWhite, Qt::AlignCenter, m_Name);
}

void QGradientRamp::resizeEvent(QResizeEvent* pResizeEvent)
{
	m_LinearGradient.setStart(rect().left(), 0.0);
	m_LinearGradient.setFinalStop(rect().right(), 0.0);
}

QGradientStops QGradientRamp::GetGradientStops(void) const
{
	return m_GradientStops;
}

void QGradientRamp::SetGradientStops(const QGradientStops& GradientStops)
{
	m_GradientStops = GradientStops;
	update();
}