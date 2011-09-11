
// Precompiled headers
#include "Stable.h"

#include "FocusWidget.h"
#include "RenderThread.h"
#include "Camera.h"

QFocusWidget::QFocusWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_FocusTypeComboBox(),
	m_FocalDistanceSlider(),
	m_FocalDistanceSpinner()
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
	
    m_FocalDistanceSpinner.setRange(0.0, 100.0);
	m_FocalDistanceSpinner.setSuffix(" mm");
	m_GridLayout.addWidget(&m_FocalDistanceSpinner, 0, 2);
	
	connect(&m_FocalDistanceSlider, SIGNAL(valueChanged(double)), &m_FocalDistanceSpinner, SLOT(setValue(double)));
	connect(&m_FocalDistanceSlider, SIGNAL(valueChanged(double)), this, SLOT(SetFocalDistance(double)));
	connect(&m_FocalDistanceSpinner, SIGNAL(valueChanged(double)), &m_FocalDistanceSlider, SLOT(setValue(double)));
	connect(&gRenderStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gRenderStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	connect(&gCamera.GetFocus(), SIGNAL(Changed(const QFocus&)), this, SLOT(OnFocusChanged(const QFocus&)));

	emit gRenderStatus.StatisticChanged("Camera", "Focus", "", "", "");
}

void QFocusWidget::SetFocalDistance(const double& FocalDistance)
{
	gCamera.GetFocus().SetFocalDistance(FocalDistance);
}

void QFocusWidget::OnRenderBegin(void)
{
	OnFocusChanged(gCamera.GetFocus());
}

void QFocusWidget::OnRenderEnd(void)
{
	gCamera.GetFocus().Reset();
}

void QFocusWidget::OnFocusChanged(const QFocus& Focus)
{
	m_FocalDistanceSlider.setValue(Focus.GetFocalDistance(), true);
	m_FocalDistanceSpinner.setValue(Focus.GetFocalDistance(), true);

	emit gRenderStatus.StatisticChanged("Focus", "Focal Distance", QString::number(Focus.GetFocalDistance(), 'f', 2), "mm");
}