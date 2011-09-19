
// Precompiled headers
#include "Stable.h"

#include "ApertureWidget.h"
#include "RenderThread.h"
#include "Camera.h"

QApertureWidget::QApertureWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_SizeSlider(),
	m_SizeSpinner()
{
	setTitle("Aperture");
	setStatusTip("Aperture properties");
	setToolTip("Aperture properties");

	setLayout(&m_GridLayout);

	// Aperture size
	m_GridLayout.addWidget(new QLabel("Size"), 3, 0);

	m_SizeSlider.setOrientation(Qt::Horizontal);
	m_SizeSlider.setRange(0.0, 0.5);
	m_GridLayout.addWidget(&m_SizeSlider, 3, 1);
	
    m_SizeSpinner.setRange(0.0, 0.5);
	m_SizeSpinner.setSuffix(" mm");
	m_GridLayout.addWidget(&m_SizeSpinner, 3, 2);
	
	connect(&m_SizeSlider, SIGNAL(valueChanged(double)), &m_SizeSpinner, SLOT(setValue(double)));
	connect(&m_SizeSpinner, SIGNAL(valueChanged(double)), &m_SizeSlider, SLOT(setValue(double)));
	connect(&m_SizeSlider, SIGNAL(valueChanged(double)), this, SLOT(SetAperture(double)));
	connect(&gRenderStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gRenderStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	connect(&gCamera.GetAperture(), SIGNAL(Changed(const QAperture&)), this, SLOT(OnApertureChanged(const QAperture&)));

	emit gRenderStatus.StatisticChanged("Camera", "Aperture", "", "", "");
}

void QApertureWidget::SetAperture(const double& Aperture)
{
	gCamera.GetAperture().SetSize(Aperture);
}

void QApertureWidget::OnRenderBegin(void)
{
	gCamera.GetAperture().Reset();
}

void QApertureWidget::OnRenderEnd(void)
{
	gCamera.GetAperture().Reset();
}

void QApertureWidget::OnApertureChanged(const QAperture& Aperture)
{
 	m_SizeSlider.setValue(Aperture.GetSize(), true);
	m_SizeSpinner.setValue(Aperture.GetSize(), true);

	emit gRenderStatus.StatisticChanged("Aperture", "Size", QString::number(Aperture.GetSize(), 'f', 3), "mm");
}