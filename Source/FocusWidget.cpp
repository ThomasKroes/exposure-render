
#include "FocusWidget.h"
#include "RenderThread.h"
#include "Camera.h"

CFocusWidget::CFocusWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_FocusTypeComboBox(),
	m_FocalDistanceSlider(),
	m_FocalDistanceSpinBox()
{
	setTitle("Focus");
	setStatusTip("Focus properties");
	setToolTip("Focus properties");

	setLayout(&m_GridLayout);

	// Focus type
//	m_GridLayout.addWidget(new QLabel("Focus type"), 5, 0);

//	m_FocusTypeComboBox.addItem("Automatic");
// 	m_FocusTypeComboBox.addItem("Pick");
// 	m_FocusTypeComboBox.addItem("Manual");
//	m_GridLayout.addWidget(&m_FocusTypeComboBox, 5, 1, 1, 2);
	
//	connect(&m_FocusTypeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(SetFocusType(int)));

	// Focal distance
	m_GridLayout.addWidget(new QLabel("Focal distance"), 0, 0);

	m_FocalDistanceSlider.setOrientation(Qt::Horizontal);
    m_FocalDistanceSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_FocalDistanceSlider.setRange(0.0, 100.0);
	m_GridLayout.addWidget(&m_FocalDistanceSlider, 0, 1);
	
    m_FocalDistanceSpinBox.setRange(0.0, 1000000.0);
	m_GridLayout.addWidget(&m_FocalDistanceSpinBox, 0, 2);
	
	connect(&m_FocalDistanceSlider, SIGNAL(valueChanged(double)), &m_FocalDistanceSpinBox, SLOT(setValue(double)));
	connect(&m_FocalDistanceSlider, SIGNAL(valueChanged(double)), this, SLOT(SetFocalDistance(double)));
	connect(&m_FocalDistanceSpinBox, SIGNAL(valueChanged(double)), &m_FocalDistanceSlider, SLOT(setValue(double)));
	connect(&gRenderStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gRenderStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	connect(&gCamera.GetFocus(), SIGNAL(Changed(const QFocus&)), this, SLOT(OnFocusChanged(const QFocus&)));
}

void CFocusWidget::SetFocalDistance(const double& FocalDistance)
{
	gCamera.GetFocus().SetFocalDistance(FocalDistance);
}

void CFocusWidget::OnRenderBegin(void)
{
	OnFocusChanged(gCamera.GetFocus());
}

void CFocusWidget::OnRenderEnd(void)
{
	gCamera.GetFocus().Reset();
}

void CFocusWidget::OnFocusChanged(const QFocus& Focus)
{
	m_FocalDistanceSlider.setValue(Focus.GetFocalDistance(), true);
}