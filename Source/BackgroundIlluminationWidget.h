#pragma once

class QLight;

class QBackgroundIlluminationWidget : public QGroupBox
{
    Q_OBJECT

public:
    QBackgroundIlluminationWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const { return QSize(10, 10); }

public slots:
	void OnBackgroundIlluminationChanged(bool Checked);
	void OnGradientColorTopChanged(const QColor& Color);
	void OnGradientColorMiddleChanged(const QColor& Color);
	void OnGradientColorBottomChanged(const QColor& Color);
	void OnTopIntensityChanged(double Intensity);
	void OnMiddleIntensityChanged(double Intensity);
	void OnBottomIntensityChanged(double Intensity);
	void OnUseTextureChanged(int UseTexture);
	void OnLoadTexture(void);
	void OnBackgroundChanged(void);

protected:
	QGridLayout			m_MainLayout;
	QColorPushButton	m_GradientColorTop;
	QDoubleSlider		m_IntensitySliderTop;
	QDoubleSpinner		m_IntensitySpinBoxTop;
	QColorPushButton	m_GradientColorMiddle;
	QDoubleSlider		m_IntensitySliderMiddle;
	QDoubleSpinner		m_IntensitySpinBoxMiddle;
	QColorPushButton	m_GradientColorBottom;
	QDoubleSlider		m_IntensitySliderBottom;
	QDoubleSpinner		m_IntensitySpinBoxBottom;
	QCheckBox			m_UseTexture;
	QLineEdit			m_TextureFilePath;
	QPushButton			m_LoadTexture;
};