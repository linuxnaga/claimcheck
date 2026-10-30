"""Cost-share and waiting-period rules (section 4 of the specification). Plain Python, no model."""
import calendar
from datetime import date
from decimal import ROUND_HALF_UP, Decimal


def rupees(x):
    """Round to whole rupees, half up."""
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def cost_share(bill, deductible, copay_pct, oop_max, room_rent_cap=None, room_days=0, room_rate=0,
               deductible_met=0, oop_used=0):
    """What the member pays and what Asteria pays on one bill.

    Room charges above the cap are never covered and do not count towards the out-of-pocket maximum.
    """
    if min(bill, deductible, copay_pct, oop_max, room_days, room_rate, deductible_met, oop_used) < 0:
        raise ValueError("amounts cannot be negative")
    room_excess = max(0, room_rate - room_rent_cap) * room_days if room_rent_cap is not None else 0
    room_excess = min(room_excess, bill)
    eligible = bill - room_excess
    deductible_part = min(eligible, max(0, deductible - deductible_met))
    copay = (eligible - deductible_part) * copay_pct / 100
    capped_share = min(deductible_part + copay, max(0, oop_max - oop_used))
    member_pays = capped_share + room_excess
    return {
        "room_excess": rupees(room_excess),
        "eligible_amount": rupees(eligible),
        "deductible_part": rupees(deductible_part),
        "copay": rupees(copay),
        "capped_share": rupees(capped_share),
        "member_pays": rupees(member_pays),
        "asteria_pays": rupees(bill) - rupees(member_pays),
    }


def add_months(d: date, months: int) -> date:
    """Same day of the month, or the last day of the month when that day does not exist."""
    m = d.month - 1 + months
    y, m = d.year + m // 12, m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def waiting_period(start: date, months: int, service: date):
    end = add_months(start, months)
    covered = service >= end
    return {"wait_ends": end.isoformat(), "covered": covered,
            "days_left": 0 if covered else (end - service).days}
