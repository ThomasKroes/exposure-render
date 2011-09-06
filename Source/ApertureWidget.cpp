
#include "ApertureWidget.h"
#include "MainWindow.h"
#include "RenderThread.h"

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

	m_ApertureSizeSlider.setOrientation(Qt::Horizontal);
    m_ApertureSizeSlider.setFocusPolicy(Qt::StrongFocus);
    m_ApertureSizeSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_ApertureSizeSlider.setRange(0.0, 1.0);
	m_GridLayout.addWidget(&m_ApertureSizeSlider, 3, 1);
	
    m_ApertureSizeSpinBox.setRange(0.0, 1.0);
	m_GridLayout.addWidget(&m_ApertureSizeSpinBox, 3, 2);
	
	connect(&m_ApertureSizeSlider, SIGNAL(valueChanged(double)), &m_ApertureSizeSpinBox, SLOT(setValue(double)));
	connect(&m_ApertureSizeSlider, SIGNAL(valueChanged(double)), this, SLOT(SetAperture(double)));
	connect(&m_ApertureSizeSpinBox, SIGNAL(valueChanged(double)), &m_ApertureSizeSlider, SLOT(setValue(double)));
}

void CApertureWidget::SetAperture(const double& Aperture)
{
	if (!Scene())
		return;

	Scene()->m_Camera.m_Aperture.m_Size = (float)Aperture;

	// Flag the camera as dirty, this will restart the rendering
	Scene()->m_DirtyFlags.SetFlag(CameraDirty);
}