#pragma once

class CStatus : public QObject
{
	Q_OBJECT

public:
	void SetRenderBegin(void);
	void SetRenderEnd(void);
	void SetPreRenderFrame(void);
	void SetPostRenderFrame(void);
	void SetResize(void);
	void SetLoadPreset(const QString& PresetName);
	void SetStatisticChanged(const QString& Group, const QString& Name, const QString& Value, const QString& Unit = "", const QString& Icon = "");
	
signals:
	void RenderBegin(void);
	void RenderEnd(void);
	void PreRenderFrame(void);
	void PostRenderFrame(void);
	void Resize(void);
	void LoadPreset(const QString& PresetName);
	void StatisticChanged(const QString& Group, const QString& Name, const QString& Value, const QString& Unit = "", const QString& Icon = "");
};

extern CStatus gStatus;