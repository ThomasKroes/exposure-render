
#include "Controls.h"

QColorPushButton::QColorPushButton(QWidget* pParent) :
QPushButton(pParent),
	m_Margin(5),
	m_Radius(9),
	m_Color(Qt::gray)
{
	setText("");
}

QSize QColorPushButton::sizeHint() const
{
	return QSize(100, 20);
}

void QColorPushButton::paintEvent(QPaintEvent* pPaintEvent)
{
	setText("");

	QPushButton::paintEvent(pPaintEvent);

	QPainter Painter(this);

	// Get button rectangle
	QRect ColorRectangle = pPaintEvent->rect();

	// Deflate it
	ColorRectangle.adjust(m_Margin, m_Margin, -(m_Margin + 75), -m_Margin);

	// Use anti aliasing
	Painter.setRenderHints(QPainter::Antialiasing);

	// Rectangle styling
	Painter.setBrush(QBrush(isEnabled() ? m_Color : Qt::gray));
	Painter.setPen(QPen(isEnabled() ? QColor(25, 25, 25) : Qt::gray, 0.7));

	// Draw
	Painter.drawRoundedRect(ColorRectangle, m_Radius, Qt::SizeMode::AbsoluteSize);

	// Move rectangle to the right
	ColorRectangle.setLeft(ColorRectangle.right() + 3);
	ColorRectangle.setWidth(rect().width() - ColorRectangle.left());

	// Draw text
	Painter.setFont(QFont("Arial", 7));
	Painter.setPen(QPen(isEnabled() ? QColor(25, 25, 25) : Qt::gray));
	Painter.drawText(ColorRectangle, Qt::AlignCenter, "[" + QString::number(m_Color.red()) + ", " + QString::number(m_Color.green()) + ", " + QString::number(m_Color.blue()) + "]");
}

void QColorPushButton::mousePressEvent(QMouseEvent* pEvent)
{
	QColorDialog ColorDialog;

	connect(&ColorDialog, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));

	ColorDialog.setWindowIcon(QIcon(":/Images/color--pencil.png"));

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

void QColorPushButton::SetColor(const QColor& Color)
{
	m_Color = Color;
	update();
}

void QColorPushButton::OnCurrentColorChanged(const QColor& Color)
{
	SetColor(Color);

	emit currentColorChanged(m_Color);
}

QFloatSlider::QFloatSlider(QWidget* pParent /*= NULL*/) :
	QSlider(pParent),
	m_Factor(10000.0f)
{

}

