#pragma once

class CRenderStatus : public QObject
{
	Q_OBJECT

public:
	void SetStatisticChanged(const QString& Group, const QString& Name, const QString& Value, const QString& Unit = "", const QString& Icon = "");

signals:
	void RenderBegin(void);
	void RenderEnd(void);
	void PreRenderFrame(void);
	void PostRenderFrame(void);
	void Resize(void);
	void LoadPreset(const QString& PresetName);
	void StatisticChanged(const QString& Group, const QString& Name, const QString& Value, const QString& Unit = "", const QString& Icon = "");
	
	friend class QRenderThread;
	friend class QFilmWidget;
	friend class QApertureWidget;
	friend class QProjectionWidget;
	friend class QFocusWidget;
};

extern CRenderStatus gRenderStatus;