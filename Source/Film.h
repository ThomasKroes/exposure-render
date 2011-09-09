#pragma once

#include <QtGui>

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
	void		ReadXML(QDomElement& Parent);
	QDomElement	WriteXML(QDomDocument& DOM, QDomElement& Parent);

signals:
	void Changed(void);

private:
	int				m_Width;
	int				m_Height;
	int				m_Dirty;
};