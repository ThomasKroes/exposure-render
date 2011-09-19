
// Precompiled headers
#include "Stable.h"

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
	Painter.drawRoundedRect(ColorRectangle, m_Radius, Qt::AbsoluteSize);

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

	ColorDialog.setWindowIcon(GetIcon("color--pencil"));

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

QDoubleSlider::QDoubleSlider(QWidget* pParent /*= NULL*/) :
	QSlider(pParent),
	m_Multiplier(1000.0)
{
	connect(this, SIGNAL(valueChanged(int)), this, SLOT(setValue(int)));

	setSingleStep(1);
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