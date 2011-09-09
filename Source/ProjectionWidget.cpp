
#include "ProjectionWidget.h"
#include "RenderThread.h"
#include "Camera.h"

CProjectionWidget::CProjectionWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_FieldOfViewSlider(),
	m_FieldOfViewSpinBox()
{
	setTitle("Projection");
	setStatusTip("Projection properties");
	setToolTip("Projection properties");

	setLayout(&m_GridLayout);

	// Field of view
	m_GridLayout.addWidget(new QLabel("Field of view"), 4, 0);

	m_FieldOfViewSlider.setOrientation(Qt::Horizontal);
	m_FieldOfViewSlider.setRange(10, 200);
	m_GridLayout.addWidget(&m_FieldOfViewSlider, 4, 1);
	
    m_FieldOfViewSpinBox.setRange(10, 200);
	m_GridLayout.addWidget(&m_FieldOfViewSpinBox, 4, 2);
	
	connect(&m_FieldOfViewSlider, SIGNAL(valueChanged(double)), &m_FieldOfViewSpinBox, SLOT(setValue(double)));
	connect(&m_FieldOfViewSlider, SIGNAL(valueChanged(double)), this, SLOT(SetFieldOfView(double)));
	connect(&m_FieldOfViewSpinBox, SIGNAL(valueChanged(double)), &m_FieldOfViewSlider, SLOT(setValue(double)));
	connect(&gRenderStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gRenderStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	connect(&gCamera.GetProjection(), SIGNAL(Changed(const QProjection&)), this, SLOT(OnProjectionChanged(const QProjection&)));
}

void CProjectionWidget::SetFieldOfView(const double& FieldOfView)
{
	gCamera.GetProjection().SetFieldOfView(FieldOfView);
}

void CProjectionWidget::OnRenderBegin(void)
{
	OnProjectionChanged(gCamera.GetProjection());
}

void CProjectionWidget::OnRenderEnd(void)
{
	gCamera.GetProjection().Reset();
}

void CProjectionWidget::OnProjectionChanged(const QProjection& Projection)
{
	m_FieldOfViewSlider.setValue(Projection.GetFieldOfView(), true);
	m_FieldOfViewSpinBox.setValue(Projection.GetFieldOfView(), true);
}