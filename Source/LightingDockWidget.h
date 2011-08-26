#pragma once

#include <QtGui>

class CLightingWidget : public QWidget
{
    Q_OBJECT

public:
    CLightingWidget(QWidget* pParent = NULL);
	void CreateActions(void);

private slots:
	void AddLight(void);
	void LockHeight(const int& State);
	void SetTheta(const int& Theta);
	void SetPhi(const int& Phi);
	void SetDistance(const int& Distance);
	void SetWidth(const int& Width);
	void SetHeight(const int& Height);

protected:
	QVBoxLayout*	m_pMainLayout;
	QGroupBox*		m_pLightsGroupBox;
	QListView*		m_pListView;
	QLineEdit*		m_pLightName;
	QPushButton*	m_pAddLight;
	QPushButton*	m_pRemoveLight;
	QPushButton*	m_pRenameLight;
	QPushButton*	m_pLoadLights;
	QPushButton*	m_pSaveLights;
	QGroupBox*		m_pLightSettingsGroupBox;
	QLabel*			m_pThetaLabel;
	QSlider*		m_pThetaSlider;
	QSpinBox*		m_pThetaSpinBox;
	QLabel*			m_pPhiLabel;
	QSlider*		m_pPhiSlider;
	QSpinBox*		m_pPhiSpinBox;
	QLabel*			m_pDistanceLabel;
	QSlider*		m_pDistanceSlider;
	QSpinBox*		m_pDistanceSpinBox;
	QLabel*			m_pWidthLabel;
	QSlider*		m_pWidthSlider;
	QSpinBox*		m_pWidthSpinBox;
	QLabel*			m_pHeightLabel;
	QSlider*		m_pHeightSlider;
	QSpinBox*		m_pHeightSpinBox;
	QCheckBox*		m_pLockHeightCheckBox;
	QLabel*			m_pIntensityLabel;
	QSlider*		m_pIntensitySlider;
	QSpinBox*		m_pIntensitySpinBox;

	// Actions
    QAction*		m_pLightNameAction;
    QAction*		m_pAddLightAction;
    QAction*		m_pRemoveLightAction;
    QAction*		m_pRenameLightAction;
	QAction*		m_pLoadLightsAction;
	QAction*		m_pSaveLightsAction;
};

class CLightingDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    CLightingDockWidget(QWidget *parent = 0);

private:
	CLightingWidget*	m_pLightingWidget;
};