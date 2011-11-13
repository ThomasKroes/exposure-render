/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "AppearanceDockWidget.h"
#include "VtkWidget.h"

QAppearanceWidget::QAppearanceWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_PresetsWidget(NULL, "Appearance", "Appearance"),
	m_AppearanceSettingsWidget(),
	m_TransferFunctionWidget(),
	m_NodeSelectionWidget(),
	m_NodePropertiesWidget()
{
	// Create main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	m_MainLayout.addWidget(&m_PresetsWidget, 0, 0);
	m_MainLayout.addWidget(&m_AppearanceSettingsWidget, 1, 0);
	m_MainLayout.addWidget(&m_TransferFunctionWidget, 2, 0);
	m_MainLayout.addWidget(&m_NodeSelectionWidget, 3, 0);
	m_MainLayout.addWidget(&m_NodePropertiesWidget, 4, 0);
	
	QObject::connect(&m_PresetsWidget, SIGNAL(LoadPreset(const QString&)), this, SLOT(OnLoadPreset(const QString&)));
	QObject::connect(&gStatus, SIGNAL(LoadPreset(const QString&)), &m_PresetsWidget, SLOT(OnLoadPreset(const QString&)));
	QObject::connect(&m_PresetsWidget, SIGNAL(SavePreset(const QString&)), this, SLOT(OnSavePreset(const QString&)));
	QObject::connect(&gTransferFunction, SIGNAL(Changed()), this, SLOT(OnTransferFunctionChanged()));
}

void QAppearanceWidget::OnLoadPreset(const QString& Name)
{
	m_PresetsWidget.LoadPreset(gTransferFunction, Name);
}

void QAppearanceWidget::OnSavePreset(const QString& Name)
{
	QTransferFunction Preset = gTransferFunction;
	Preset.SetName(Name);

	// Save the preset
	m_PresetsWidget.SavePreset(Preset);
}

void QAppearanceWidget::OnTransferFunctionChanged(void)
{
	if (!gpActiveRenderWidget || !gpActiveRenderWidget->GetVolumeMapper()->GetInput())
		return;

	vtkSmartPointer<vtkPiecewiseFunction> Opacity		= vtkPiecewiseFunction::New();
	vtkSmartPointer<vtkPiecewiseFunction> DiffuseR		= vtkPiecewiseFunction::New();
	vtkSmartPointer<vtkPiecewiseFunction> DiffuseG		= vtkPiecewiseFunction::New();
	vtkSmartPointer<vtkPiecewiseFunction> DiffuseB		= vtkPiecewiseFunction::New();
	vtkSmartPointer<vtkPiecewiseFunction> SpecularR		= vtkPiecewiseFunction::New();
	vtkSmartPointer<vtkPiecewiseFunction> SpecularG		= vtkPiecewiseFunction::New();
	vtkSmartPointer<vtkPiecewiseFunction> SpecularB		= vtkPiecewiseFunction::New();
	vtkSmartPointer<vtkPiecewiseFunction> Glossiness	= vtkPiecewiseFunction::New();
	vtkSmartPointer<vtkPiecewiseFunction> EmissionR		= vtkPiecewiseFunction::New();
	vtkSmartPointer<vtkPiecewiseFunction> EmissionG		= vtkPiecewiseFunction::New();
	vtkSmartPointer<vtkPiecewiseFunction> EmissionB		= vtkPiecewiseFunction::New();

	double* pScalarRange = gpActiveRenderWidget->GetVolumeMapper()->GetInput()->GetScalarRange();

	for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
	{
		const double Intensity = pScalarRange[0] + gTransferFunction.GetNode(i).GetIntensity() * (pScalarRange[1] - pScalarRange[0]);

		Opacity->AddPoint(Intensity, gTransferFunction.GetNode(i).GetOpacity());
		DiffuseR->AddPoint(Intensity, gTransferFunction.GetNode(i).GetDiffuse().redF());
		DiffuseG->AddPoint(Intensity, gTransferFunction.GetNode(i).GetDiffuse().greenF());
		DiffuseR->AddPoint(Intensity, gTransferFunction.GetNode(i).GetDiffuse().blueF());
		SpecularR->AddPoint(Intensity, gTransferFunction.GetNode(i).GetSpecular().redF());
		SpecularG->AddPoint(Intensity, gTransferFunction.GetNode(i).GetSpecular().greenF());
		SpecularR->AddPoint(Intensity, gTransferFunction.GetNode(i).GetSpecular().blueF());

		const float Gloss = 1.0f - expf(-gTransferFunction.GetNode(i).GetGlossiness());

		Glossiness->AddPoint(Intensity, Gloss * 250.0);
		EmissionR->AddPoint(Intensity, 100.0f * gTransferFunction.GetNode(i).GetEmission().redF());
		EmissionG->AddPoint(Intensity, 100.0f * gTransferFunction.GetNode(i).GetEmission().greenF());
		EmissionR->AddPoint(Intensity, 100.0f * gTransferFunction.GetNode(i).GetEmission().blueF());
	}
		
	gpActiveRenderWidget->GetVolumeProperty()->SetOpacity(Opacity);
	gpActiveRenderWidget->GetVolumeProperty()->SetDiffuse(0, DiffuseR);
	gpActiveRenderWidget->GetVolumeProperty()->SetDiffuse(1, DiffuseG);
	gpActiveRenderWidget->GetVolumeProperty()->SetDiffuse(2, DiffuseB);
	gpActiveRenderWidget->GetVolumeProperty()->SetSpecular(0, SpecularR);
	gpActiveRenderWidget->GetVolumeProperty()->SetSpecular(1, SpecularG);
	gpActiveRenderWidget->GetVolumeProperty()->SetSpecular(2, SpecularB);
	gpActiveRenderWidget->GetVolumeProperty()->SetGlossiness(Glossiness);
	gpActiveRenderWidget->GetVolumeProperty()->SetEmission(0, EmissionR);
	gpActiveRenderWidget->GetVolumeProperty()->SetEmission(1, EmissionG);
	gpActiveRenderWidget->GetVolumeProperty()->SetEmission(2, EmissionB);

	gpActiveRenderWidget->GetVolumeProperty()->SetDensityScale(gTransferFunction.GetDensityScale());
	gpActiveRenderWidget->GetVolumeProperty()->SetGradientFactor(gTransferFunction.GetGradientFactor());
	gpActiveRenderWidget->GetVolumeProperty()->SetShadingType(gTransferFunction.GetShadingType());
}

QAppearanceDockWidget::QAppearanceDockWidget(QWidget *parent) :
	QDockWidget(parent),
	m_VolumeAppearanceWidget()
{
	setWindowTitle("Appearance");
	setToolTip("<img src=':/Images/palette.png'><div>Volume Appearance</div>");
	setWindowIcon(GetIcon("palette"));

	m_VolumeAppearanceWidget.setParent(this);

	setWidget(&m_VolumeAppearanceWidget);
}