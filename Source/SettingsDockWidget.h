#pragma once

#include "Controls.h"

class CTracerSettingsWidget : public QGroupBox
{
    Q_OBJECT

public:
    CTracerSettingsWidget(QWidget* pParent = NULL);

private slots:
	void SetNoBounces(const int& NoBounces);
	void SetPhase(const int& Phase);

private:
	QGridLayout		m_MainLayout;
	QComboBox		m_RenderTypeComboBox;
	QSlider			m_NoBouncesSlider;
	QSpinBox		m_NoBouncesSpinBox;
	QGridLayout		m_ScatteringLayout;
	QDoubleSlider	m_PhaseSlider;
};

class CKernelSettingsWidget : public QGroupBox
{
    Q_OBJECT

public:
    CKernelSettingsWidget(QWidget* pParent = NULL);

private slots:
	void SetKernelWidth(const int& KernelWidth);
	void SetKernelHeight(const int& KernelHeight);
	void LockKernelHeight(const int& Lock);

private:
	QGridLayout		m_MainLayout;
	QDoubleSlider	m_KernelWidthSlider;
	QSpinBox		m_KernelWidthSpinBox;
	QDoubleSlider	m_KernelHeightSlider;
	QSpinBox		m_KernelHeightSpinBox;
	QCheckBox		m_LockKernelHeightCheckBox;
};

class CSettingsWidget : public QWidget
{
    Q_OBJECT

public:
    CSettingsWidget(QWidget* pParent = NULL);

private:
	QGridLayout				m_MainLayout;
	CTracerSettingsWidget	m_TracerSettingsWidget;
	CKernelSettingsWidget	m_KernelSettingsWidget;
};

class QSettingsDockWidget : public QDockWidget
{
	Q_OBJECT

public:
    QSettingsDockWidget(QWidget* pParent = NULL);

private:
	
	CSettingsWidget		m_SettingsWidget;
};