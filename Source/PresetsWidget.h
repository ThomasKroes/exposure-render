#pragma once

#include <QtGui>
#include <QtXml\qdom.h>

#include "Preset.h"
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

template <class T>
class QPresets
{
public:
	void Add(T Preset)
	{
		m_Presets.append(Preset);
	}

	void Remove(T Preset)
	{
		m_Presets.remove(Preset);
	}

	void LoadPresetsFromFile(const bool& ChoosePath = false)
	{

	}

	void SavePresetsFromFile(const bool& ChoosePath = false)
	{

	}

protected slots:

protected:
	QList<T>		m_Presets;
};

class QPresetsWidget : public QGroupBox
{
    Q_OBJECT

public:
	QPresetsWidget(const QString& PresetFileName = "", QWidget* pParent = NULL);

	void CreateUI(void);

	void CreateConnections(void);

	void LoadPresetsFromFile(const bool& ChoosePath = false);

	void SavePresetsToFile(const bool& ChoosePath = false);

	void LoadPresets(QDomElement& Root);

	void SavePresets(QDomDocument& DomDoc, QDomElement& Root);

	void LoadPreset(QPresetXML* pPreset);

	void SavePreset(const QString& Name);

	void UpdatePresetsList(void);

	void OnLoadPreset(void);

	void OnSavePreset(void);

	void OnRemovePreset(void);

	void OnLoadPresets(void);

	void OnSavePresets(void);

	void OnPresetNameChanged(const QString& Text);

	void OnPresetItemChanged(QListWidgetItem* pWidgetItem);

	void OnApplicationAboutToExit(void);

	virtual QSize sizeHint() const { return QSize(10, 10); }

protected slots:

protected:
	QString			m_PresetFileName;
	QGridLayout		m_MainLayout;
	QComboBox		m_PresetName;
	QPushButton		m_LoadPreset;
	QPushButton		m_SavePreset;
	QPushButton		m_RemovePreset;
	QPushButton		m_LoadPresets;
	QPushButton		m_SavePresets;
	QPresetList		m_PresetItems;
};

class QTestWidget : public QPushButton
{
	Q_OBJECT

public:
	QTestWidget(void)
	{
		qDebug("QTestWidget::SavePresets");

		connect(this, SIGNAL(clicked()), this, SLOT(OnSavePresets()));

	}

	virtual void SavePresets(void)
	{

	};

protected:

private:

public slots:

	void OnSavePresets(void)
	{
		this->SavePresets();
	}

	signals:
};

template <class T>
class QTemplateWidget : public QTestWidget
{
public:
	QTemplateWidget(void) :
		QTestWidget()
	{
	}

	void SavePresets(void)
	{
		qDebug("QTemplateWidget::SavePresets");
	};

	QList<T>	m_Presets;
};