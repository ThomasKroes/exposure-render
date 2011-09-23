
// Precompiled headers
#include "Stable.h"

#include "TransferFunctionWidget.h"
#include "TransferFunction.h"

QTransferFunctionWidget::QTransferFunctionWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_TopLeftLayout(),
	m_TopLayout(),
	m_TopRightLayout(),
	m_LeftLayout(),
	m_MiddleLayout(),
	m_RightLayout(),
	m_BottomLeftLayout(),
	m_BottomLayout(),
	m_BottomRightLayout(),
	m_Canvas(),
	m_GradientRampDiffuse("Diffuse"),
	m_GradientRampSpecular("Specular"),
	m_GradientRampEmission("Emission")
{
	// Set the size policy, making sure the widget fits nicely in the layout
	setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

	// Title, status and tooltip
	setTitle("Transfer Function");
	setToolTip("Transfer function properties");
	setStatusTip("Transfer function properties");

	setLayout(&m_MainLayout);

	m_MainLayout.setContentsMargins(5, 5, 5, 15);
	m_MainLayout.setSpacing(10);

	m_MainLayout.addLayout(&m_TopLeftLayout, 0, 0);
	m_MainLayout.addLayout(&m_TopLayout, 0, 1);
	m_MainLayout.addLayout(&m_TopRightLayout, 0, 2);

	m_MainLayout.addLayout(&m_LeftLayout, 1, 0);
	m_MainLayout.addLayout(&m_MiddleLayout, 1, 1);
	m_MainLayout.addLayout(&m_RightLayout, 2, 2);

	m_MainLayout.addLayout(&m_BottomLeftLayout, 2, 0);
	m_MainLayout.addLayout(&m_BottomLayout, 2, 1);
	m_MainLayout.addLayout(&m_BottomRightLayout, 2, 2);

	m_MiddleLayout.addWidget(&m_Canvas);

	m_Canvas.setContentsMargins(0, 0, 0, 0);

	m_BottomLayout.setContentsMargins(25, 0, 15, 0);

	QGradientStops DiffuseGradientStops, SpecularGradientStops, EmissionGradientStops;
	
	DiffuseGradientStops << QGradientStop(0, QColor(100, 100, 100, 50)) << QGradientStop(1, QColor::fromHsl(30, 150, 100));
	SpecularGradientStops << QGradientStop(0, QColor(100, 100, 100, 50)) << QGradientStop(1, QColor::fromHsl(130, 150, 100));
	EmissionGradientStops << QGradientStop(0, QColor(100, 100, 100, 50)) << QGradientStop(1, QColor::fromHsl(200, 150, 50));

	m_GradientRampDiffuse.SetGradientStops(DiffuseGradientStops);
	m_GradientRampSpecular.SetGradientStops(SpecularGradientStops);
	m_GradientRampEmission.SetGradientStops(EmissionGradientStops);

	m_GradientRampDiffuse.setFixedHeight(18);
	m_GradientRampSpecular.setFixedHeight(18);
	m_GradientRampEmission.setFixedHeight(18);

	m_BottomLayout.addWidget(&m_GradientRampDiffuse, 0, 0);
	m_BottomLayout.addWidget(&m_GradientRampSpecular, 1, 0);
 	m_BottomLayout.addWidget(&m_GradientRampEmission, 2, 0);

	QObject::connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	QObject::connect(&gStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	QObject::connect(&gTransferFunction, SIGNAL(Changed()), this, SLOT(OnUpdateGradients()));
}

void QTransferFunctionWidget::OnRenderBegin(void)
{
	m_Canvas.setEnabled(true);
	m_Canvas.SetHistogram(gHistogram);
}

void QTransferFunctionWidget::OnRenderEnd(void)
{
	m_Canvas.setEnabled(false);
	m_Canvas.SetHistogram(QHistogram());
}

void QTransferFunctionWidget::OnUpdateGradients(void)
{
	QGradientStops GradientStopsDiffuse, GradientStopsSpecular, GradientStopsEmission;

	for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
	{
		QColor Diffuse, Specular, Emission;

		Diffuse		= gTransferFunction.GetNode(i).GetDiffuse();
		Specular	= gTransferFunction.GetNode(i).GetSpecular();
		Emission	= gTransferFunction.GetNode(i).GetEmission();

		Diffuse.setAlphaF(gTransferFunction.GetNode(i).GetOpacity());

		GradientStopsDiffuse << QGradientStop(gTransferFunction.GetNode(i).GetIntensity(), Diffuse);
		GradientStopsSpecular << QGradientStop(gTransferFunction.GetNode(i).GetIntensity(), Specular);
		GradientStopsEmission << QGradientStop(gTransferFunction.GetNode(i).GetIntensity(), Emission);
	}

	m_GradientRampDiffuse.SetGradientStops(GradientStopsDiffuse);
	m_GradientRampSpecular.SetGradientStops(GradientStopsSpecular);
	m_GradientRampEmission.SetGradientStops(GradientStopsEmission);
}