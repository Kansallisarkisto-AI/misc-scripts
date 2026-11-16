from __future__ import annotations

import random
from calendar import monthrange
from datetime import date
from typing import Callable, Sequence, Mapping
from math import gcd
from dataclasses import dataclass
import math


CHECK_KEYS = "0123456789ABCDEFHJKLMNPRSTUVWXY"

CENTURY_SEPARATORS = {
    18: "+",
    19: "-",
    20: "A",
}

DEFAULT_DATE_FORMATS = (
    # Common contemporary Finnish/Swedish numeric formats
    "{day}.{month}.{year}",       # 5.7.2026
    "{day2}.{month2}.{year}",     # 05.07.2026
    "{day}/{month}/{year}",       # 5/7/2026
    "{day2}/{month2}/{year}",     # 05/07/2026
    "{year}-{month2}-{day2}",     # 2026-07-05

    # Short-year and compact historical-style variants
    "{day}.{month}.{year2}",      # 5.7.26
    "{day2}.{month2}.{year2}",    # 05.07.26
    "{day}/{month}-{year2}",      # 5/7-26
    "{day2}/{month2}-{year2}",    # 05/07-26
    "{day2}{month2}{year}",       # 05072026
    "{day2}{month2}{year2}",       # 050726
)


def sample_time(
    rng: random.Random,
    *,
    separator: str = ":",
    pad_hour: bool = True,
) -> str:
    """Generate a random 24-hour clock time."""
    hour = rng.randrange(24)
    minute = rng.randrange(60)

    hour_text = f"{hour:02d}" if pad_hour else str(hour)
    return f"{hour_text}{separator}{minute:02d}"


def sample_numerical_date(
    rng: random.Random,
    *,
    start_year: int = 1600,
    end_year: int | None = None,
    formats: Sequence[str] = DEFAULT_DATE_FORMATS,
) -> str:
    """
    Generate a valid random date using a randomly selected Finnish/Swedish
    numerical date format.
    """
    if end_year is None:
        end_year = date.today().year

    if start_year > end_year:
        raise ValueError("start_year cannot be greater than end_year")

    if not formats:
        raise ValueError("formats cannot be empty")

    year = rng.randint(start_year, end_year)
    month = rng.randint(1, 12)
    day = rng.randint(1, monthrange(year, month)[1])

    template = rng.choice(formats)
    return template.format(
        day=day,
        day2=f"{day:02d}",
        month=month,
        month2=f"{month:02d}",
        year=year,
        year2=f"{year % 100:02d}",
    )


def sample_integer(
    rng: random.Random,
    *,
    min_digits: int = 1,
    max_digits: int = 7,
    leading_zeros: bool = True,
    lam=0.1
) -> str:
    """Generate a random integer-like string."""
    if not 1 <= min_digits <= max_digits:
        raise ValueError("Expected 1 <= min_digits <= max_digits")

    #digits = rng.randint(min_digits, max_digits)
    weights = [math.exp(-lam * (i - min_digits)) for i in range(min_digits, max_digits+1)]
    digits = rng.choices(
            population=range(min_digits, max_digits+1),
            weights=weights,
            k=1
        )[0]

    if leading_zeros:
        return "".join(str(rng.randrange(10)) for _ in range(digits))

    if digits == 1:
        return str(rng.randrange(10))

    return str(rng.randrange(1, 10)) + "".join(
        str(rng.randrange(10)) for _ in range(digits - 1)
    )


def sample_decimal(
    rng: random.Random,
    *,
    min_total_digits: int = 1,
    max_total_digits: int = 6,
    min_decimal_places: int = 0,
    max_decimal_places: int = 4,
    separators: Sequence[str] = (",", "."),
    lam=0.1
) -> str:
    """
    Generate a decimal string allowing a single leading zero.

    The integer part is either ``0`` or begins with a nonzero digit. Values
    with multiple leading zeros, such as ``00.5`` and ``012.3``, are not
    generated.

    The number of total digits excludes the decimal separator. A separator is
    included even when decimal_places is zero, so values such as ``15.``,
    ``7,``, and ``0.`` are possible.
    """
    if not 1 <= min_total_digits <= max_total_digits:
        raise ValueError(
            "Expected 1 <= min_total_digits <= max_total_digits"
        )

    if not 0 <= min_decimal_places <= max_decimal_places:
        raise ValueError(
            "Expected 0 <= min_decimal_places <= max_decimal_places"
        )

    if not separators:
        raise ValueError("separators cannot be empty")

    possible_decimal_places = [
        places
        for places in range(min_decimal_places, max_decimal_places + 1)
        if places < max_total_digits
    ]

    if not possible_decimal_places:
        raise ValueError(
            "The configured digit limits leave no room for an integer part"
        )

    #decimal_places = rng.choice(possible_decimal_places)

    first_place = possible_decimal_places[0]

    weights = [math.exp(-lam * (i - first_place)) for i in possible_decimal_places]

    decimal_places = rng.choices(
        population=possible_decimal_places,
        weights=weights,
        k=1
    )[0]

    minimum_digits = max(min_total_digits, decimal_places + 1)

    #total_digits = rng.randint(minimum_digits, max_total_digits)

    weights = [math.exp(-lam * (i - 1)) for i in range(minimum_digits, max_total_digits+1)]

    total_digits = rng.choices(
        population=range(minimum_digits, max_total_digits+1),
        weights=weights,
        k=1
    )[0]

    integer_digits = total_digits - decimal_places

    if integer_digits == 1:
        integer_part = str(rng.randrange(10))
    else:
        integer_part = str(rng.randrange(1, 10)) + "".join(
            str(rng.randrange(10))
            for _ in range(integer_digits - 1)
        )

    fractional_part = "".join(
        str(rng.randrange(10))
        for _ in range(decimal_places)
    )

    separator = rng.choice(separators)
    return f"{integer_part}{separator}{fractional_part}"


def sample_hetu(
    rng: random.Random,
    *,
    start_year: int = 1850,
    end_year: int = 2040,
    min_individual_number: int = 2,
    max_individual_number: int = 899,
) -> str:
    """Generate a syntactically valid Finnish personal identity code."""
    if start_year > end_year:
        raise ValueError("start_year cannot be greater than end_year")

    unsupported_years = [
        year
        for year in (start_year, end_year)
        if year // 100 not in CENTURY_SEPARATORS
    ]
    if unsupported_years:
        raise ValueError(
            "HETU generation supports years 1800 through 2099"
        )

    if not 2 <= min_individual_number <= max_individual_number <= 899:
        raise ValueError(
            "Individual number must be between 002 and 899"
        )

    year = rng.choices(
        range(start_year, end_year + 1),
        weights=[5 if 1900 <= y <= 1999 else 1 for y in range(start_year, end_year + 1)],
        k=1,
    )[0]  # emphasize 1900s in Finnish identity number
    month = rng.randint(1, 12)
    day = rng.randint(1, monthrange(year, month)[1])
    individual_number = rng.randint(
        min_individual_number,
        max_individual_number,
    )

    century_separator = CENTURY_SEPARATORS[year // 100]
    year_short = year % 100

    check_number = int(
        f"{day:02d}{month:02d}{year_short:02d}{individual_number:03d}"
    )
    check_key = CHECK_KEYS[check_number % 31]

    return (
        f"{day:02d}{month:02d}{year_short:02d}"
        f"{century_separator}{individual_number:03d}{check_key}"
    )

def sample_fraction(
    rng: random.Random,
    *,
    max_numerator: int = 15,
    max_denominator: int = 16,
    allow_improper: bool = False,
    allow_mixed: bool = True,
    max_whole_part: int = 20,
) -> str:
    """
    Generate an ASCII-only fraction such as:

        1/2
        5/16
        2 1/4

    Fractions are reduced to lowest terms.
    """
    if max_numerator < 1:
        raise ValueError("max_numerator must be at least 1")

    if max_denominator < 2:
        raise ValueError("max_denominator must be at least 2")

    if max_whole_part < 1:
        raise ValueError("max_whole_part must be at least 1")

    denominator = rng.randint(2, max_denominator)

    if allow_improper:
        numerator = rng.randint(1, max_numerator)
    else:
        numerator = rng.randint(
            1,
            min(max_numerator, denominator - 1),
        )

    common_divisor = gcd(numerator, denominator)
    numerator //= common_divisor
    denominator //= common_divisor

    fraction = f"{numerator}/{denominator}"

    if allow_mixed and rng.choice((False, True)):
        whole_part = rng.randint(1, max_whole_part)
        return f"{whole_part} {fraction}"

    return fraction

def sample_year_range(
    rng: random.Random,
    *,
    start_year: int = 1600,
    end_year: int = 2026,
    min_span: int = 1,
    max_span: int = 20,
    separators: Sequence[str] = ("-", "/"),
    allow_abbreviated_end: bool = True,
) -> str:
    """
    Generate an ASCII-only year range such as:

        1872-1875
        1799/1800
        1882-85
    """
    if start_year > end_year:
        raise ValueError("start_year cannot be greater than end_year")

    if not 1 <= min_span <= max_span:
        raise ValueError("Expected 1 <= min_span <= max_span")

    if not separators:
        raise ValueError("separators cannot be empty")

    if any(not separator.isascii() for separator in separators):
        raise ValueError("All separators must contain ASCII characters only")

    latest_start = end_year - min_span
    if latest_start < start_year:
        raise ValueError(
            "Year range is too small for the requested minimum span"
        )

    first_year = rng.randint(start_year, latest_start)
    maximum_possible_span = min(max_span, end_year - first_year)
    span = rng.randint(min_span, maximum_possible_span)
    second_year = first_year + span

    separator = rng.choice(separators)

    if (
        allow_abbreviated_end
        and first_year // 100 == second_year // 100
        and rng.choice((False, True))
    ):
        second_text = f"{second_year % 100:02d}"
    else:
        second_text = str(second_year)

    return f"{first_year}{separator}{second_text}"

DEFAULT_CURRENCY_UNITS: dict[str, tuple[str, ...]] = {
    "mk": ("p",),
    "mark": ("p",),
    "kr": ("ore",),
    "rdr": ("sk",),
}


def sample_currency_amount(
    rng: random.Random,
    *,
    max_major_units: int = 9999,
    currency_units: Mapping[str, Sequence[str]] = DEFAULT_CURRENCY_UNITS,
    formats: Sequence[str] = (
        "major_only",
        "decimal",
        "colon",
        "major_minor",
    ),
) -> str:
    """
    Generate an ASCII-only archival currency amount.

    Examples:

        15 mk
        15,50 mk
        15:50
        2 kr 50 ore
        12 rdr
        8 rdr 16 sk

    Major and minor units are selected from compatible pairs defined by
    ``currency_units``.
    """
    if max_major_units < 0:
        raise ValueError("max_major_units cannot be negative")

    if not currency_units:
        raise ValueError("currency_units cannot be empty")

    if not formats:
        raise ValueError("formats cannot be empty")

    allowed_formats = {
        "major_only",
        "decimal",
        "colon",
        "major_minor",
    }
    unknown_formats = set(formats) - allowed_formats

    if unknown_formats:
        raise ValueError(
            f"Unknown currency formats: {sorted(unknown_formats)}"
        )

    for currency, minor_units in currency_units.items():
        if not currency:
            raise ValueError("Currency labels cannot be empty")

        if not currency.isascii():
            raise ValueError(
                f"Currency label must be ASCII: {currency!r}"
            )

        if not minor_units:
            raise ValueError(
                f"Currency {currency!r} must have at least one minor unit"
            )

        for minor_unit in minor_units:
            if not minor_unit:
                raise ValueError("Minor-unit labels cannot be empty")

            if not minor_unit.isascii():
                raise ValueError(
                    f"Minor-unit label must be ASCII: {minor_unit!r}"
                )

    major = rng.randint(0, max_major_units)
    minor = rng.randint(0, 99)
    currency = rng.choice(tuple(currency_units))
    selected_format = rng.choice(tuple(formats))

    if selected_format == "major_only":
        return f"{major} {currency}"

    if selected_format == "decimal":
        separator = rng.choice((",", "."))
        return f"{major}{separator}{minor:02d} {currency}"

    if selected_format == "colon":
        return f"{major}:{minor:02d}"

    compatible_minor_units = currency_units[currency]
    minor_unit = rng.choice(tuple(compatible_minor_units))

    return f"{major} {currency} {minor} {minor_unit}"

def sample_record_number(
    rng: random.Random,
    *,
    minimum: int = 1,
    maximum: int = 9999,
    prefixes: Sequence[str] = (
        "No ",
        "No. ",
        "N:o ",
        "N:r ",
        "Nr ",
        "Nro ",
    ),
) -> str:
    """
    Generate an ASCII-only labelled record number such as:

        No 14
        No. 14
        N:o 7
        Nr 302
    """
    if not 1 <= minimum <= maximum:
        raise ValueError("Expected 1 <= minimum <= maximum")

    if not prefixes:
        raise ValueError("prefixes cannot be empty")

    if any(not prefix.isascii() for prefix in prefixes):
        raise ValueError("All prefixes must contain ASCII characters only")

    return f"{rng.choice(prefixes)}{rng.randint(minimum, maximum)}"

def _to_roman(number: int) -> str:
    """Convert an integer from 1 through 3999 to a Roman numeral."""
    if not 1 <= number <= 3999:
        raise ValueError("Roman numeral value must be between 1 and 3999")

    symbols = (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    )

    result: list[str] = []

    for value, symbol in symbols:
        count, number = divmod(number, value)
        result.append(symbol * count)

    return "".join(result)


def sample_roman_numeral(
    rng: random.Random,
    *,
    minimum: int = 1,
    maximum: int = 30,
    lowercase_probability: float = 0.0,
    suffixes: Sequence[str] = ("", ".", ":"),
) -> str:
    """
    Generate a Roman numeral such as ``IV``, ``XII.`` or ``xv:``.
    """
    if not 1 <= minimum <= maximum <= 3999:
        raise ValueError("Expected 1 <= minimum <= maximum <= 3999")

    if not 0.0 <= lowercase_probability <= 1.0:
        raise ValueError("lowercase_probability must be between 0 and 1")

    if not suffixes:
        raise ValueError("suffixes cannot be empty")

    numeral = _to_roman(rng.randint(minimum, maximum))

    if rng.random() < lowercase_probability:
        numeral = numeral.lower()

    return numeral + rng.choice(suffixes)


def sample_case_number(
    rng: random.Random,
    *,
    start_year: int = 1600,
    end_year: int = 2026,
    max_sequence_number: int = 9999,
    formats: Sequence[str] = (
        "{number}/{year}",
        "{year}:{number}",
        "{number}/{roman}/{year}",
        "{number}-{year}",
    ),
) -> str:
    """
    Generate a case or diary number such as:

        14/1873
        1873:14
        12/II/1902
        271-1954
    """
    if start_year > end_year:
        raise ValueError("start_year cannot be greater than end_year")

    if max_sequence_number < 1:
        raise ValueError("max_sequence_number must be positive")

    if not formats:
        raise ValueError("formats cannot be empty")

    year = rng.randint(start_year, end_year)
    number = rng.randint(1, max_sequence_number)
    roman = _to_roman(rng.randint(1, 12))

    return rng.choice(formats).format(
        number=number,
        year=year,
        year2=f"{year % 100:02d}",
        roman=roman,
    )

def sample_special_sequence(
    rng: random.Random,
    *,
    characters=('"', "'", ".", "..", "...", ",", ",,", ",,,", "-", "-'-", '-"-'),
) -> str:
    """
    Sample a special ascii character sequence from a list
    """

    return rng.choice(characters)

@dataclass(frozen=True)
class DomainGenerator:
    generator: Callable[[random.Random], str]
    weight: float = 1.0


DOMAIN_GENERATORS: dict[str, DomainGenerator] = {
    "time": DomainGenerator(sample_time, weight=0.333),
    "date": DomainGenerator(sample_numerical_date, weight=0.333),
    "integer": DomainGenerator(sample_integer, weight=3.0),
    "decimal": DomainGenerator(sample_decimal, weight=3.0),
    #"hetu": DomainGenerator(sample_hetu, weight=0.5),
    #"fraction": DomainGenerator(sample_fraction, weight=0.1),
    "yearrange": DomainGenerator(sample_year_range, weight=0.333),
    #"currency": DomainGenerator(sample_currency_amount, weight=1.0),
    "recordnumber": DomainGenerator(sample_record_number, weight=0.333),
    "romannumeral": DomainGenerator(sample_roman_numeral, weight=0.333),
    #"casenumber": DomainGenerator(sample_case_number, weight=0.1),
    "specialsequence": DomainGenerator(sample_special_sequence, weight=0.1),
}


def sample_numeral(
    rng: random.Random,
    *,
    domains: Sequence[str] | None = None,
) -> str:
    """Select a weighted domain and generate a string from that domain."""
    available_domains = tuple(
        DOMAIN_GENERATORS if domains is None else domains
    )

    if not available_domains:
        raise ValueError("At least one domain must be enabled")

    unknown_domains = set(available_domains) - DOMAIN_GENERATORS.keys()
    if unknown_domains:
        raise ValueError(
            f"Unknown numeral domains: {sorted(unknown_domains)}"
        )

    weights = [
        DOMAIN_GENERATORS[domain].weight
        for domain in available_domains
    ]

    if any(weight < 0 for weight in weights):
        raise ValueError("Domain weights cannot be negative")

    if not any(weights):
        raise ValueError("At least one enabled domain must have a positive weight")

    domain = rng.choices(
        available_domains,
        weights=weights,
        k=1,
    )[0]

    return DOMAIN_GENERATORS[domain].generator(rng)