#pragma once

class QGradientRamp : public QWidget
{
public:
	QGradientRamp(const QString& Name, QWidget* pParent = NULL);

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
	QColor				m_TextForegroundEnabled;
	QColor				m_TextForegroundDisabled;
	QColor				m_TextBackgroundEnabled;
	QColor				m_TextBackgroundDisabled;
};