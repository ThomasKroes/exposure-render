
#include "BackgroundIlluminationWidget.h"
#include "LightsWidget.h"
#include "RenderThread.h"

QBackgroundIlluminationWidget::QBackgroundIlluminationWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_Color(),
	m_ColorLayout(),
	m_IntensitySlider(),
	m_IntensitySpinBox(),
	m_UseTexture(),
	m_TextureFilePath(),
	m_LoadTexture()
{
	// Title, status and tooltip
	setTitle("Background Illumination");
	setToolTip("Background Illumination");
	setStatusTip("Background Illumination");

	// Allow user to turn background illumination on/off
	setCheckable(true);

	// Apply main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	// Color
	m_MainLayout.addWidget(new QLabel("Color"), 0, 0);
	m_MainLayout.addWidget(&m_Color, 0, 1);

	m_MainLayout.addLayout(&m_ColorLayout, 0, 2);

	m_ColorLayout.addWidget(new QLabel("Intensity"), 0, 0);

	m_IntensitySlider.setOrientation(Qt::Orientation::Horizontal);
	m_IntensitySlider.setFocusPolicy(Qt::StrongFocus);
	m_IntensitySlider.setTickPosition(QSlider::TickPosition::NoTicks);
	m_IntensitySlider.setRange(0, 100);
	m_ColorLayout.addWidget(&m_IntensitySlider, 0, 1);

	m_IntensitySpinBox.setRange(0, 100);
	m_MainLayout.addWidget(&m_IntensitySpinBox, 0, 3);

	// Use Texture
	m_MainLayout.addWidget(&m_UseTexture, 1, 1);

	// Texture
	m_MainLayout.addWidget(new QLabel("Texture"), 2, 0);

	// Path
	m_TextureFilePath.setFixedHeight(22);
	m_MainLayout.addWidget(&m_TextureFilePath, 2, 1, 1, 2);

	m_LoadTexture.setIcon(QIcon(":/Images/folder-open-image.png"));
	m_LoadTexture.setFixedWidth(22);
	m_LoadTexture.setFixedHeight(22);
	m_MainLayout.addWidget(&m_LoadTexture, 2, 3);

//	connect(&m_ThetaSlider, SIGNAL(valueChanged(int)), &m_ThetaSpinBox, SLOT(setValue(int)));
//	connect(&m_ThetaSpinBox, SIGNAL(valueChanged(int)), &m_ThetaSlider, SLOT(setValue(int)));
//	connect(&m_ThetaSlider, SIGNAL(valueChanged(int)), this, SLOT(OnThetaChanged(int)));
}