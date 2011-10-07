
// Precompiled headers
#include "Stable.h"

#include "ProjectionWidget.h"
#include "RenderThread.h"
#include "Camera.h"

QProjectionWidget::QProjectionWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_FieldOfViewSlider(),
	m_FieldOfViewSpinner()
{
	setTitle("Projection");
	setStatusTip("Projection properties");
	setToolTip("Projection properties");

	m_GridLayout.setColumnMinimumWidth(0, 75);
	setLayout(&m_GridLayout);

	// Field of view
	m_GridLayout.addWidget(new QLabel("Field of view"), 4, 0);

	m_FieldOfViewSlider.setOrientation(Qt::Horizontal);
	m_FieldOfViewSlider.setRange(10.0, 150.0);
	m_GridLayout.addWidget(&m_FieldOfViewSlider, 4, 1);
	
    m_FieldOfViewSpinner.setRange(10.0, 150.0);
	m_FieldOfViewSpinner.setSuffix(" deg.");
	m_GridLayout.addWidget(&m_FieldOfViewSpinner, 4, 2);
	
	connect(&m_FieldOfViewSlider, SIGNAL(valueChanged(double)), &m_FieldOfViewSpinner, SLOT(setValue(double)));
	connect(&m_FieldOfViewSlider, SIGNAL(valueChanged(double)), this, SLOT(SetFieldOfView(double)));
	connect(&m_FieldOfViewSpinner, SIGNAL(valueChanged(double)), &m_FieldOfViewSlider, SLOT(setValue(double)));
	connect(&gCamera.GetProjection(), SIGNAL(Changed(const QProjection&)), this, SLOT(OnProjectionChanged(const QProjection&)));

	gStatus.SetStatisticChanged("Camera", "Projection", "", "", "");
}

void QProjectionWidget::SetFieldOfView(const double& FieldOfView)
{
	gCamera.GetProjection().SetFieldOfView(FieldOfView);
}

void QProjectionWidget::OnProjectionChanged(const QProjection& Projection)
{
	m_FieldOfViewSlider.setValue(Projection.GetFieldOfView(), true);
	m_FieldOfViewSpinner.setValue(Projection.GetFieldOfView(), true);

	gStatus.SetStatisticChanged("Projection", "Field Of View", QString::number(Projection.GetFieldOfView(), 'f', 2), "Deg.");
}