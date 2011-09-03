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
	
protected:
	QGridLayout			m_MainLayout;
	QColorPushButton	m_Color;
	QGridLayout			m_ColorLayout;
	QSlider				m_IntensitySlider;
	QSpinBox			m_IntensitySpinBox;
	QCheckBox			m_UseTexture;
	QLineEdit			m_TextureFilePath;
	QPushButton			m_LoadTexture;
};