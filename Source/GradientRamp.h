#pragma once

class QGradientRamp : public QWidget
{
public:
	QGradientRamp(const QString& Name = "Gradient Ramp", QWidget* pParent = NULL);

	void				paintEvent(QPaintEvent * pe);
	virtual void		resizeEvent(QResizeEvent* pResizeEvent);

	QGradientStops		GetGradientStops(void) const;
	void				SetGradientStops(const QGradientStops& GradientStops);

private:
	QString				m_Name;
	QGradientStops		m_GradientStops;
	QLinearGradient		m_LinearGradient;
	QColor				m_CheckerDimEnabled;
	QColor				m_CheckerDimDisabled;
	QColor				m_CheckerBrightEnabled;
	QColor				m_CheckerBrightDisabled;
	QPen				m_PenEnabled;
	QPen				m_PenDisabled;
	QFont				m_Font;
	QPen				m_TextEnabled;
	QPen				m_TextDisabled;
};