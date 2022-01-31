import re
from asr_language_model_evaluation.preprocessing.num2words_wrapper import num2words

REMOVE_URL = re.compile(r'http\S+')
HTML_TAG = re.compile(r'<[^>]*>')
MULTIPLE_SPACES = re.compile(r'\s+')
NUMBERS = re.compile(r'([0-9]+(\.[0-9]+)*)')
PERCENTAGE = re.compile(r'\%')
PUNCTUATION = re.compile(r'[\!\"\#\$\%\&\'\(\)\*\+,-.\/:;\<=\>\?\@\[\\\]\^\_\`\{\|\}\~]')
SUPERSCRIPT = re.compile(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]')
MONEY_FULL = re.compile(r'(R\$(\s)*[0-9.]+(,[0-9]+)*)')
MONEY = re.compile(r'R\$')
HOURS = re.compile(r'[0-9]+h')
HOURS_24_PATTERN = re.compile(r'[0-9]{1,2}[h:][0-9]{1,2}')
HOURS_24_SEPARATOR = re.compile(r'[h:]')
METRICS = re.compile(r'(([0-9]+\s*)((c?m)([²³23])?))')
DATE_PATTERN = re.compile(r'([0-9]{1,2})\/([0-9]{1,2})(\/([0-9]{4,4}|[0-9]{2,2})){0,1}')
APOSTROPHE = re.compile(r'(\s|^)([a-z])\'([a-z]+)(\s|$)')

FILLED_PAUSES = {
    'ah': re.compile(r'(\s|^)(ã)(\s|$)'),
    'eh': re.compile(r'(\s|^)(éh|ehm|ehn)(\s|$)'),
    'uh': re.compile(r'(\s|^)(hum|hm|uhm)(\s|$)')
}


def normalize_filled_pauses(text: str) -> str:
    for standard, pattern in FILLED_PAUSES.items():
        text = pattern.sub(f' {standard} ', text)
    return text


def normalize_numbers(text: str) -> str:
    for match in NUMBERS.findall(text):
        number_expansion = num2words(match[0].replace('.', ''))
        text = text.replace(match[0], f' {number_expansion} ')
    return text


def get_money_name(value: int, decimal: bool = False):
    if decimal:
        expansion = 'centavo'
        if value != 1:
            expansion = f'{expansion}s'
        return expansion

    expansion = 'real'
    if value != 1:
        expansion = f'{expansion[:-1]}is'
    return expansion


def transform_money(money: str) -> str:
    number = MONEY.sub(' ', money).strip().split(',')
    if number[0].replace('.', '') == '':
        return ''
    integer_part = int(number[0].replace('.', ''))
    decimal_part = int(number[1]) if len(number) > 1 else None
    in_full = num2words(integer_part)
    in_full = f'{in_full} {get_money_name(integer_part)}'
    if integer_part == 0 and len(number) > 1 and int(decimal_part) > 0:
        cents_in_full = num2words(decimal_part)
        return f'{cents_in_full} {get_money_name(decimal_part, True)}'

    if len(number) > 1 and int(decimal_part) > 0:
        cents_in_full = num2words(decimal_part)
        in_full = f'{in_full} e {cents_in_full} {get_money_name(decimal_part, True)}'
    return in_full


def normalize_money(text):
    all_money = MONEY_FULL.findall(text)
    for money in all_money:
        money = money[0]
        in_full = transform_money(money)
        text = text.replace(money, in_full)
    return text


def normalize_hours(text):
    all_hours = HOURS.findall(text)
    for hour in all_hours:
        number_in_full = num2words(int(hour[:-1]), 'female')
        text = text.replace(hour, f' {number_in_full} horas ')
    return text


def normalize_24h(text):
    all_timestamp = HOURS_24_PATTERN.findall(text)
    for timestamp in all_timestamp:
        hours, minutes = HOURS_24_SEPARATOR.split(timestamp)
        str_time = f'{num2words(hours, "female")} horas'
        if int(minutes) > 0:
            str_time = f'{str_time} e {num2words(minutes)} minutos'
        text = text.replace(timestamp, str_time)
    return text


def get_metrics_name(acronym: str) -> str:
    if acronym == 'cm':
        return 'centimetros'
    if acronym == 'm':
        return 'metros'
    raise Exception(f'Invalid metric acronym: {acronym}')


METRIC_POWER = {
    '2': 'quadrados',
    '3': 'cúbicos',
    '²': 'quadrados',
    '³': 'cúbicos'
}


def normalize_metrics(text: str) -> str:
    all_metrics = METRICS.findall(text)
    for metric in all_metrics:
        metric_name = get_metrics_name(metric[3])
        metric_power = METRIC_POWER.get(metric[4], '')
        text = text.replace(metric[0], f'{metric[1]} {metric_name} {metric_power} ')

    return text


def normalize_date(text):
    out = []
    for token in text.split(' '):
        match = DATE_PATTERN.match(token)
        if match is None:
            out.append(token)
            continue
        str_date = f'{num2words(match.group(1))} do {num2words(match.group(2))}'
        year = match.group(4)
        if year:
            str_date = f'{str_date} de {year}'
        out.append(str_date)
    return ' '.join(out)


def normalize(text: str) -> str:
    # remove anything from web
    text = REMOVE_URL.sub(' ', text)
    text = HTML_TAG.sub(' ', text)

    # fix apostrophe
    text = APOSTROPHE.sub(r' \2\3 ', text)

    # money fix
    text = normalize_money(text)

    # normalize hours
    text = normalize_24h(text)
    text = normalize_hours(text)

    # normalize date
    text = normalize_date(text)

    # normalize metric
    text = normalize_metrics(text)

    # lower case
    text = text.lower()

    text = PERCENTAGE.sub(' porcento ', text)

    # acronyms

    # filled pauses
    text = normalize_filled_pauses(text)

    # num2words
    text = normalize_numbers(text)

    text = PUNCTUATION.sub(' ', text)
    text = SUPERSCRIPT.sub(' ', text)

    text = NUMBERS.sub(' ', text)
    # multiple spaces
    text = MULTIPLE_SPACES.sub(' ', text)

    return text.strip()
