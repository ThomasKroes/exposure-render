#pragma once

#include <QtGui>
#include <QtXml\qdom.h>

#include "TransferFunction.h"

class QPresetItem : public QListWidgetItem
{
public:
	QPresetItem(QListWidget* pListWidget, const QString& Name, void* pData) :
		QListWidgetItem(pListWidget),
		m_pData(pData)
	{
		setText(Name);
	}

	void* m_pData;
};

class QPresetsWidget : public QGroupBox
{
    Q_OBJECT

public:
    QPresetsWidget(const QString& PresetFileName, QWidget* pParent = NULL);
	virtual ~QPresetsWidget(void);

	virtual QSize sizeHint() const { return QSize(10, 10); }

	virtual void LoadPresetsFromFile(const bool& ChoosePath = false);
	virtual void SavePresetsToFile(const bool& ChoosePath = false);
	virtual void LoadPresets(QDomElement& Root) = 0;
	virtual void SavePresets(QDomDocument& DomDoc, QDomElement& Root) = 0;
	virtual void LoadPreset(QPresetXML* pPreset);
	virtual void SavePreset(const QString& Name);

protected slots:
	virtual void CreateConnections(void);
	virtual void CreateUI(void);
	virtual void UpdatePresetsList(void);
	virtual void OnLoadPresets(void);
	virtual void OnSavePresets(void);
	virtual void OnLoadPreset(void);
	virtual void OnSavePreset(void);
	virtual void OnRemovePreset(void);
	virtual void OnDummy(void);
	virtual void OnPresetNameChanged(const QString& Text);
	virtual void OnPresetItemChanged(QListWidgetItem* pWidgetItem);

protected:
	QString			m_PresetFileName;
	QGridLayout		m_MainLayout;
	QComboBox		m_PresetName;
	QPushButton		m_LoadPreset;
	QPushButton		m_SavePreset;
	QPushButton		m_RemovePreset;
	QPushButton		m_LoadPresets;
	QPushButton		m_SavePresets;
	QPresetList		m_Presets;
};