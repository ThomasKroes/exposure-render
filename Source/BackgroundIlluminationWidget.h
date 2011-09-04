#pragma once

#include <QtGui>

#include "Controls.h"

class QLight;

class QBackgroundIlluminationWidget : public QGroupBox
{
    Q_OBJECT

public:
    QBackgroundIlluminationWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const { return QSize(10, 10); }

public slots:
	void OnBackgroundIlluminationChanged(bool Checked);
	void OnColorChanged(const QColor& Color);
	void SetIntensity(double Intensity);
	void OnUseTextureChanged(int UseTexture);
	void OnLoadTexture(void);
	void OnBackgroundChanged(void);

protected:
	QGridLayout			m_MainLayout;
	QColorPushButton	m_Color;
	QDoubleSlider		m_IntensitySlider;
	QDoubleSpinner		m_IntensitySpinBox;
	QCheckBox			m_UseTexture;
	QLineEdit			m_TextureFilePath;
	QPushButton			m_LoadTexture;
};