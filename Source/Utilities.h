#pragma once

class QMargin
{
public:
	QMargin(void) :
		m_Left(0),
		m_Right(0),
		m_Top(0),
		m_Bottom(0)
	{
	}

	QMargin(const int& Left, const int& Right, const int& Top, const int& Bottom) :
		m_Left(Left),
		m_Right(Right),
		m_Top(Top),
		m_Bottom(Bottom)
	{
	}

	int		GetLeft(void)					{ return m_Left;		}
	void	SetLeft(const int& Left)		{ m_Left = Left;		}
	int		GetRight(void)					{ return m_Right;		}
	void	SetRight(const int& Right)		{ m_Right = Right;		}
	int		GetTop(void)					{ return m_Top;			}
	void	SetTop(const int& Top)			{ m_Top = Top;			}
	int		GetBottom(void)					{ return m_Bottom;		}
	void	SetBottom(const int& Bottom)	{ m_Bottom = Bottom;	}

private:
	int		m_Left;
	int		m_Right;
	int		m_Top;
	int		m_Bottom;
};