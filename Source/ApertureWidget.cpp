
#include "ApertureWidget.h"
#include "MainWindow.h"
#include "Scene.h"

CApertureWidget::CApertureWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_ApertureSizeSlider(),
	m_ApertureSizeSpinBox()
{
	setTitle("Aperture");
	setStatusTip("Aperture properties");
	setToolTip("Aperture properties");

	setLayout(&m_GridLayout);

	// Aperture size
	m_GridLayout.addWidget(new QLabel("Size"), 3, 0);

	m_ApertureSizeSlider.setOrientation(Qt::Orientation::Horizontal);
    m_ApertureSizeSlider.setFocusPolicy(Qt::StrongFocus);
    m_ApertureSizeSlider.setTickPosition(QDoubleSlider::TickPosition::NoTicks);
	m_GridLayout.addWidget(&m_ApertureSizeSlider, 3, 1);
	
    m_ApertureSizeSpinBox.setRange(-100, 100);
	m_GridLayout.addWidget(&m_ApertureSizeSpinBox, 3, 2);
	
	connect(&m_ApertureSizeSlider, SIGNAL(valueChanged(int)), &m_ApertureSizeSpinBox, SLOT(setValue(int)));
	connect(&m_ApertureSizeSlider, SIGNAL(valueChanged(int)), this, SLOT(SetAperture(int)));
	connect(&m_ApertureSizeSpinBox, SIGNAL(valueChanged(int)), &m_ApertureSizeSlider, SLOT(setValue(int)));
}

void CApertureWidget::SetAperture(const int& Aperture)
{
	if (!gpScene)
		return;

	gpScene->m_Camera.m_Aperture.m_Size = 0.01f * (float)Aperture;

	// Flag the camera as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(CameraDirty);
}