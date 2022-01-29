import re
import traceback

from num2words import lang_PT_BR, lang_PT


def num2words(number, gender='male'):
    try:
        converter = lang_PT_BR.Num2Word_PT_BR()
        if isinstance(number, str):
            number = converter.str_to_number(number)
        result = lang_PT.Num2Word_EU.to_cardinal(converter, number)
        if gender == 'female':
            result = result.replace('um', 'uma')
            result = result.replace('dois', 'duas')

        if len(re.findall(r'\be\b', result)) == 1:
            return result

        # Transforms "mil E cento e catorze reais" into "mil, cento e catorze
        # reais"
        for ext in (
                'mil', 'milhão', 'milhões', 'bilhão', 'bilhões',
                'trilhão', 'trilhões', 'quatrilhão', 'quatrilhões'):
            if re.match('.*{} e \\w*ento'.format(ext), result):
                result = result.replace(
                    '{} e'.format(ext), '{},'.format(ext), 1
                )

        return result
    except Exception:
        traceback.print_exc()
        return ''