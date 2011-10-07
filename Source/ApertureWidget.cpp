
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

	m_GridLayout.setColumnMinimumWidth(0, 75);
	setLayout(&m_GridLayout);

	// Aperture size
	m_GridLayout.addWidget(new QLabel("Size"), 3, 0);

	m_SizeSlider.setOrientation(Qt::Horizontal);
	m_SizeSlider.setRange(0.0, 0.1);
	m_GridLayout.addWidget(&m_SizeSlider, 3, 1);
	
    m_SizeSpinner.setRange(0.0, 0.1);
	m_SizeSpinner.setSuffix(" mm");
	m_SizeSpinner.setDecimals(3);
	m_GridLayout.addWidget(&m_SizeSpinner, 3, 2);
	
	connect(&m_SizeSlider, SIGNAL(valueChanged(double)), &m_SizeSpinner, SLOT(setValue(double)));
	connect(&m_SizeSpinner, SIGNAL(valueChanged(double)), &m_SizeSlider, SLOT(setValue(double)));
	connect(&m_SizeSlider, SIGNAL(valueChanged(double)), this, SLOT(SetAperture(double)));
	connect(&gCamera.GetAperture(), SIGNAL(Changed(const QAperture&)), this, SLOT(OnApertureChanged(const QAperture&)));

	gStatus.SetStatisticChanged("Camera", "Aperture", "", "", "");
}

void QApertureWidget::SetAperture(const double& Aperture)
{
	gCamera.GetAperture().SetSize(Aperture);
}

void QApertureWidget::OnApertureChanged(const QAperture& Aperture)
{
 	m_SizeSlider.setValue(Aperture.GetSize(), true);
	m_SizeSpinner.setValue(Aperture.GetSize(), true);

	gStatus.SetStatisticChanged("Aperture", "Size", QString::number(Aperture.GetSize(), 'f', 3), "mm");
}