#pragma once

#include <QtGui>

#include "PresetsWidget.h"

class QTransferFunctionPresetsWidget : public QPresetsWidget
{
    Q_OBJECT

public:
    QTransferFunctionPresetsWidget(QWidget* pParent = NULL);
	
	virtual void LoadPresets(QDomElement& Root);
	virtual void SavePresets(QDomDocument& DomDoc, QDomElement& Root);
	virtual void LoadPreset(QPresetXML* pPreset);
	virtual void SavePreset(const QString& Name);

protected slots:
	/*
	void CreateConnections(void);
	void CreateUI(void);
	void UpdatePresetsList(void);
	void OnPresetSelectionChanged(void);
	void OnLoadPreset(void);
	void OnRemovePreset(void);
	void OnSavePreset(void);
	void OnLoadPresets(void);
	void OnSavePresets(void);
	void OnDummy(void);
	void OnPresetNameChanged(const QString& Text);
	void OnPresetItemChanged(QListWidgetItem* pWidgetItem);
	*/

protected:
	/*
	QGridLayout				m_MainLayout;
	QLineEdit				m_PresetName;
	QPushButton				m_LoadPreset;
	QPushButton				m_SavePreset;
	QPushButton				m_RemovePreset;
	QPushButton				m_LoadPresets;
	QPushButton				m_SavePresets;
	QPushButton				m_Dummy;
	QListWidget				m_PresetList;
	QTransferFunctionList	m_TransferFunctions;
	*/
};