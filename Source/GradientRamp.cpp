
// Precompiled headers
#include "Stable.h"

#include "GradientRamp.h"

QGradientRamp::QGradientRamp(const QString& Name, QWidget* pParent /*= NULL*/) :
	QWidget(pParent),
	m_Name(Name.toLower()),
	m_GradientStops(),
	m_LinearGradient(),
	m_CheckerDimEnabled(QColor::fromHsl(0, 0, 140)),
	m_CheckerDimDisabled(QColor::fromHsl(0, 0, 170)),
	m_CheckerBrightEnabled(QColor::fromHsl(0, 0, 255)),
	m_CheckerBrightDisabled(QColor::fromHsl(0, 0, 230)),
	m_PenEnabled(QColor::fromHsl(0, 0, 100)),
	m_PenDisabled(QColor::fromHsl(0, 0, 150)),
	m_Font("Arial"),
	m_TextForegroundEnabled(QColor::fromHsl(0, 0, 250)),
	m_TextForegroundDisabled(QColor::fromHsl(0, 0, 230)),
	m_TextBackgroundEnabled(QColor::fromHsl(0, 0, 20)),
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