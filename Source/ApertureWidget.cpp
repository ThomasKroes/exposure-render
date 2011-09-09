
#include "ApertureWidget.h"
#include "RenderThread.h"
#include "Camera.h"

CApertureWidget::CApertureWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_SizeSlider(),
	m_SizeSpinBox()
{
	setTitle("Aperture");
	setStatusTip("Aperture properties");
	setToolTip("Aperture properties");

	setLayout(&m_GridLayout);

	// Aperture size
	m_GridLayout.addWidget(new QLabel("Size"), 3, 0);

	m_SizeSlider.setOrientation(Qt::Horizontal);
    m_SizeSlider.setFocusPolicy(Qt::StrongFocus);
    m_SizeSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_SizeSlider.setRange(0.0, 1.0);
	m_GridLayout.addWidget(&m_SizeSlider, 3, 1);
	
    m_SizeSpinBox.setRange(0.0, 1.0);
	m_GridLayout.addWidget(&m_SizeSpinBox, 3, 2);
	
	connect(&m_SizeSlider, SIGNAL(valueChanged(double)), &m_SizeSpinBox, SLOT(setValue(double)));
	connect(&m_SizeSlider, SIGNAL(valueChanged(double)), this, SLOT(SetAperture(double)));
	connect(&m_SizeSpinBox, SIGNAL(valueChanged(double)), &m_SizeSlider, SLOT(setValue(double)));
	connect(&gRenderStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gRenderStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	connect(&gCamera.GetAperture(), SIGNAL(Changed(const QAperture&)), this, SLOT(OnApertureChanged(const QAperture&)));
}

void CApertureWidget::SetAperture(const double& Aperture)
{
	gCamera.GetAperture().SetSize(Aperture);
}

void CApertureWidget::OnRenderBegin(void)
{
	OnApertureChanged(gCamera.GetAperture());
}

void CApertureWidget::OnRenderEnd(void)
{
	gCamera.GetAperture().Reset();
}

void CApertureWidget::OnApertureChanged(const QAperture& Aperture)
{
	m_SizeSlider.setValue(Aperture.GetSize(), true);
	m_SizeSpinBox.setValue(Aperture.GetSize(), true);
}
