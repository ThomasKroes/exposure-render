#pragma once

#include "Preset.h"

class QFilm : public QPresetXML
{
	Q_OBJECT

public:
	QFilm(QObject* pParent = NULL);
	QFilm::QFilm(const QFilm& Other);
	QFilm& QFilm::operator=(const QFilm& Other);

	int			GetWidth(void) const;
	void		SetWidth(const int& Width);
	int			GetHeight(void) const;
	void		SetHeight(const int& Height);
	float		GetExposure(void) const;
	void		SetExposure(const float& Exposure);
	bool		GetNoiseReduction(void) const;
	void		SetNoiseReduction(const bool& NoiseReduction);
	void		Reset(void);
	bool		IsDirty(void) const;
	void		UnDirty(void);
	void		ReadXML(QDomElement& Parent);
	QDomElement	WriteXML(QDomDocument& DOM, QDomElement& Parent);

signals:
	void Changed(const QFilm& Film);

private:
	int			m_Width;
	int			m_Height;
	float		m_Exposure;
	bool		m_NoiseReduction;
	int			m_Dirty;
};