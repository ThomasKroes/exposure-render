#pragma once

class QLight;

class QBackgroundWidget : public QGroupBox
{
    Q_OBJECT

public:
    QBackgroundWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const { return QSize(10, 10); }

public slots:
	void OnBackgroundIlluminationChanged(bool Checked);
	void OnGradientColorTopChanged(const QColor& Color);
	void OnGradientColorMiddleChanged(const QColor& Color);
	void OnGradientColorBottomChanged(const QColor& Color);
	void OnIntensityChanged(double Intensity);
	void OnUseTextureChanged(int UseTexture);
	void OnLoadTexture(void);
	void OnBackgroundChanged(void);

protected:
	QGridLayout			m_MainLayout;
	QLabel				m_GradientColorTopLabel;
	QColorSelector		m_GradientColorTop;
	QLabel				m_GradientColorMiddleLabel;
	QColorSelector		m_GradientColorMiddle;
	QLabel				m_GradientColorBottomLabel;
	QColorSelector		m_GradientColorBottom;
	QLabel				m_IntensityLabel;
	QDoubleSlider		m_IntensitySlider;
	QDoubleSpinner		m_IntensitySpinner;
	QCheckBox			m_UseTexture;
	QLineEdit			m_TextureFilePath;
	QPushButton			m_LoadTexture;
};