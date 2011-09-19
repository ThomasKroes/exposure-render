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
	void	OnCurrentColorChanged(const QColor& Color);

signals:
	void currentColorChanged(const QColor&);

private:
	int		m_Margin;
	int		m_Radius;
	QColor	m_Color;
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