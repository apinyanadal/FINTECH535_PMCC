"""
ric_codec.py -- build and parse LSEG/OPRA option RICs (same codec as HW1/HW2).

    {ROOT}{MONTH}{DD}{YY}{STRIKE}.U{^MYY if expired}

  ROOT    equity RIC before the dot                    AMZN.O -> AMZN
  MONTH   one letter = expiry month AND right          calls A..L (Jan..Dec), puts M..X
  DD      expiry day, ZERO PADDED (HW2 finding: '...H7...' is rejected, '...H07...' works)
  YY      two-digit expiry year
  STRIKE  strike x 100, 5 digits                       230 -> 23000, 232.5 -> 23250
  .U      OPRA
  ^MYY    EXPIRED contracts only. A live contract (e.g. the Jan 2027 LEAP) has no suffix.
          HW1 finding: for puts the suffix uses the CALL letter of that month.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Optional

CALL_LETTERS = "ABCDEFGHIJKL"
PUT_LETTERS = "MNOPQRSTUVWX"


def strike_code(strike: float) -> str:
    return f"{int(round(float(strike) * 100)):05d}"


def ric_for_expiry(root: str, expiry: dt.date, strike: float, right: str = "C", expired: bool = True) -> str:
    right = right.upper()
    base = (CALL_LETTERS if right == "C" else PUT_LETTERS)[expiry.month - 1]
    yy = expiry.year % 100
    ric = f"{root.upper()}{base}{expiry.day:02d}{yy:02d}{strike_code(strike)}.U"
    if expired:
        ric += f"^{CALL_LETTERS[expiry.month - 1]}{yy:02d}"
    return ric


_RX = re.compile(r"^(?P<root>[A-Z]+?)(?P<code>[A-X])(?P<day>\d{2})(?P<yy>\d{2})(?P<strike>\d{5,6})\.U"
                 r"(?:\^(?P<sfx>[A-L])(?P<sfxyy>\d{2}))?$")


def parse_option_ric(ric: str) -> Optional[dict]:
    m = _RX.match(str(ric).strip())
    if not m:
        return None
    code = m.group("code")
    right, month = ("C", CALL_LETTERS.index(code) + 1) if code in CALL_LETTERS else ("P", PUT_LETTERS.index(code) + 1)
    try:
        expiry = dt.date(2000 + int(m.group("yy")), month, int(m.group("day")))
    except ValueError:
        return None
    return dict(root=m.group("root"), right=right, expiry=expiry, strike=int(m.group("strike")) / 100.0,
                expired=m.group("sfx") is not None)
