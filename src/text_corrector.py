"""Автокорректировка текста, полученного от системы распознавания речи.

Модуль не заменяет языковую модель и не «придумывает» слова: он исправляет
типичные ошибки распознавания (в том числе слитное написание и разговорные
формы), приводит в порядок пробелы, знаки препинания и заглавные буквы.
Оригинальный смысл фразы при этом сохраняется.

Использование::

    from src.text_corrector import correct_speech_text
    text = correct_speech_text('превет канешно щас всё будет')

Модуль намеренно не зависит от тяжёлых библиотек, чтобы его можно было
вызывать и в потоке распознавания, и в потоке анализа файла.
"""

from __future__ import annotations

import re
from functools import lru_cache


# ---------------------------------------------------------------------------
# Словарь типичных ошибок распознавания и разговорных форм.
# Ключи — в нижнем регистре; замена выполняется по границам слов.
# ---------------------------------------------------------------------------
REPLACEMENTS = {
    # Приветствия и этикет
    'превет': 'привет',
    'привед': 'привет',
    'прив ет': 'привет',
    'здраствуй': 'здравствуй',
    'здраствуйте': 'здравствуйте',
    'здравствуйти': 'здравствуйте',
    'здраствуйти': 'здравствуйте',
    'здрасьте': 'здравствуйте',
    'досвидания': 'до свидания',
    'досвиданья': 'до свидания',
    'до свиданья': 'до свидания',
    'до свидание': 'до свидания',
    'пожалуста': 'пожалуйста',
    'пожалуйсто': 'пожалуйста',
    'пожалусто': 'пожалуйста',
    'пожайлуста': 'пожалуйста',
    'пожайлусто': 'пожалуйста',
    'спасиба': 'спасибо',
    'спосибо': 'спасибо',
    'спасиба большое': 'спасибо большое',
    'извени': 'извини',
    'извените': 'извините',
    'извинити': 'извините',
    'добрый ден': 'добрый день',
    'привет как дела': 'привет, как дела',
    'привет как ты': 'привет, как ты',
    'здравствуйте как дела': 'здравствуйте, как дела',
    'добрый день как дела': 'добрый день, как дела',
    'как дила': 'как дела',

    # Частые разговорные/слитные формы
    'щас': 'сейчас',
    'щяс': 'сейчас',
    'счас': 'сейчас',
    'сйчас': 'сейчас',
    'че': 'что',
    'чё': 'что',
    'шо': 'что',
    'што': 'что',
    'чо': 'что',
    'канешно': 'конечно',
    'канешн': 'конечно',
    'конешн': 'конечно',
    'седня': 'сегодня',
    'сёдня': 'сегодня',
    'севодня': 'сегодня',
    'ваще': 'вообще',
    'вааще': 'вообще',
    'вобще': 'вообще',
    'вообщем': 'в общем',
    'вобщем': 'в общем',
    'вопщем': 'в общем',
    'исчё': 'ещё',
    'ищо': 'ещё',
    'ещо': 'ещё',
    'тока': 'только',
    'токо': 'только',
    'почиму': 'почему',
    'потомучто': 'потому что',
    'потомушта': 'потому что',
    'ничево': 'ничего',
    'ничо': 'ничего',
    'ничё': 'ничего',
    'каво': 'кого',
    'чево': 'чего',
    'чиво': 'чего',
    'кагда': 'когда',
    'тагда': 'тогда',
    'вчира': 'вчера',
    'завтро': 'завтра',
    'кажится': 'кажется',
    'кажеться': 'кажется',
    'кажиться': 'кажется',
    'наверна': 'наверное',
    'наверно': 'наверное',
    'кароче': 'короче',
    'короч': 'короче',
    'впринципе': 'в принципе',
    'впринцепе': 'в принципе',
    'впринципи': 'в принципе',

    # Дефисное написание
    'по русски': 'по-русски',
    'порусски': 'по-русски',
    'по английски': 'по-английски',
    'поанглийски': 'по-английски',
    'во первых': 'во-первых',
    'вопервых': 'во-первых',
    'во вторых': 'во-вторых',
    'вовторых': 'во-вторых',
    'в третьих': 'в-третьих',
    'втретьих': 'в-третьих',
    'наконец то': 'наконец-то',
    'наконецто': 'наконец-то',
    'как то': 'как-то',
    'както': 'как-то',
    'что то': 'что-то',
    'чтото': 'что-то',
    'кто то': 'кто-то',
    'ктото': 'кто-то',
    'где то': 'где-то',
    'гдето': 'где-то',
    'когда то': 'когда-то',
    'когдато': 'когда-то',
    'почему то': 'почему-то',
    'почемуто': 'почему-то',
    'какой то': 'какой-то',
    'какойто': 'какой-то',
    'какая то': 'какая-то',
    'какаято': 'какая-то',
    'какое то': 'какое-то',
    'какоето': 'какое-то',
    'чем то': 'чем-то',
    'чемто': 'чем-то',
    'что нибудь': 'что-нибудь',
    'чтонибудь': 'что-нибудь',
    'кто нибудь': 'кто-нибудь',
    'ктонибудь': 'кто-нибудь',
    'как нибудь': 'как-нибудь',
    'какнибудь': 'как-нибудь',
    'где нибудь': 'где-нибудь',
    'гденибудь': 'где-нибудь',
    'когда нибудь': 'когда-нибудь',
    'когданибудь': 'когда-нибудь',
    'всё таки': 'всё-таки',
    'всётаки': 'всё-таки',
    'всетаки': 'всё-таки',
    'всёравно': 'всё равно',
    'всеровно': 'всё равно',
    'тоесть': 'то есть',
    'таккак': 'так как',
    'какбудто': 'как будто',
    'по прежнему': 'по-прежнему',
    'попрежнему': 'по-прежнему',
    'по другому': 'по-другому',
    'подругому': 'по-другому',
    'по настоящему': 'по-настоящему',
    'понастоящему': 'по-настоящему',
    'по видимому': 'по-видимому',
    'повидимому': 'по-видимому',
    'чуть чуть': 'чуть-чуть',
    'чутьчуть': 'чуть-чуть',
    'кое как': 'кое-как',
    'коекак': 'кое-как',
    'кое что': 'кое-что',
    'коечто': 'кое-что',
    'из за': 'из-за',
    'изза': 'из-за',
    'из под': 'из-под',
    'изпод': 'из-под',
    'давным давно': 'давным-давно',
    'давнымдавно': 'давным-давно',
    'еле еле': 'еле-еле',
    'елееле': 'еле-еле',

    # Слитное написание «не» и других частых сочетаний
    'незнаю': 'не знаю',
    'немогу': 'не могу',
    'нехочу': 'не хочу',
    'небуду': 'не буду',
    'небыло': 'не было',
    'несмогу': 'не смогу',
    'непонимаю': 'не понимаю',
    'неслышу': 'не слышу',
    'невижу': 'не вижу',
    'непонял': 'не понял',
    'непоняла': 'не поняла',
    'неуверен': 'не уверен',
    'ненадо': 'не надо',
    'ненравится': 'не нравится',
    'мнекажется': 'мне кажется',
    'мнекажеться': 'мне кажется',
    'янезнаю': 'я не знаю',
    'янехочу': 'я не хочу',
    'янемогу': 'я не могу',
    'тычто': 'ты что',
    'этоже': 'это же',
    'втом': 'в том',
    'вэтом': 'в этом',
    'наэтом': 'на этом',
    'приэтом': 'при этом',
    'кэтому': 'к этому',
    'ктому': 'к тому',
    'такчто': 'так что',
    'нипричем': 'ни при чём',
    'нипричём': 'ни при чём',

    # Прочие характерные ослышки
    'понел': 'понял',
    'такоэ': 'такое',
    'этово': 'этого',
    'тово': 'того',
    'всево': 'всего',
    'сваво': 'своего',
    'моево': 'моего',
    'тваво': 'твоего',
    'нашево': 'нашего',
    'вашево': 'вашего',
    'ищё': 'ещё',
    'отквы': 'открой',
    'отклея': 'открой',
    'распадование': 'распознавание',
    'распадования': 'распознавания',
    'распознование': 'распознавание',
    'распознования': 'распознавания',
    'с днём рожденья': 'с днём рождения',

    # Буква «ё»: Whisper и другие движки часто пишут её как «е».
    'еще': 'ещё',
    'пойдем': 'пойдём',
    'идет': 'идёт',
    'придет': 'придёт',
    'уйдет': 'уйдёт',
    'найдет': 'найдёт',
    'зайдет': 'зайдёт',
    'войдет': 'войдёт',
    'перейдет': 'перейдёт',
    'живет': 'живёт',
    'зовет': 'зовёт',
    'поет': 'поёт',
    'пьет': 'пьёт',
    'дает': 'даёт',
    'сдает': 'сдаёт',
    'продает': 'продаёт',
    'создает': 'создаёт',
    'признает': 'признаёт',
    'встает': 'встаёт',
    'достает': 'достаёт',
    'устает': 'устаёт',
    'остается': 'остаётся',
    'удается': 'удаётся',
    'берет': 'берёт',
    'врет': 'врёт',
    'ждет': 'ждёт',
    'растет': 'растёт',
    'цветет': 'цветёт',
    'шел': 'шёл',
    'пошел': 'пошёл',
    'пришел': 'пришёл',
    'ушел': 'ушёл',
    'нашел': 'нашёл',
    'зашел': 'зашёл',
    'вошел': 'вошёл',
    'перешел': 'перешёл',
    'прошел': 'прошёл',
    'подошел': 'подошёл',
    'отошел': 'отошёл',
    'ученый': 'учёный',
    'ученые': 'учёные',
    'зеленый': 'зелёный',
    'черный': 'чёрный',
    'желтый': 'жёлтый',
    'тяжелый': 'тяжёлый',
    'легкий': 'лёгкий',
    'теплый': 'тёплый',
    'темный': 'тёмный',
    'веселый': 'весёлый',
    'дешевый': 'дешёвый',
    'жесткий': 'жёсткий',
}


# Слова, после которых в начале предложения обычно ставится запятая.
# Список намеренно ограничен, чтобы не ломать обычные фразы.
INTRO_WORDS = (
    'здравствуйте',
    'здравствуй',
    'добрый день',
    'доброе утро',
    'добрый вечер',
    'спокойной ночи',
    'привет',
    'пожалуйста',
    'конечно',
    'наверное',
    'возможно',
    'короче',
    'например',
    'кстати',
    'во-первых',
    'во-вторых',
    'в-третьих',
    'итак',
    'следовательно',
    'по-моему',
    'к сожалению',
    'к счастью',
    'кроме того',
    'более того',
    'таким образом',
    'впрочем',
    'значит',
)

# Служебные слова, повтор которых почти всегда является ошибкой распознавания.
_DUPLICATE_WORDS = (
    'в', 'во', 'на', 'и', 'а', 'но', 'что', 'это', 'с', 'со', 'к', 'ко',
    'у', 'о', 'об', 'от', 'до', 'по', 'за', 'из', 'не', 'ни', 'же', 'ли', 'бы',
)

# Сокращения, которые нельзя разбивать пробелами после точек.
_ABBREVIATIONS = (
    'т.д.', 'т.п.', 'т.е.', 'т.к.', 'т.н.', 'др.', 'г.', 'гг.', 'ул.', 'им.',
    'рис.', 'стр.', 'см.', 'напр.', 'руб.', 'коп.', 'млн.', 'млрд.', 'тыс.',
    'проф.', 'акад.', 'доц.', 'канд.', 'тех.', 'эл.', 'поч.', 'тел.', 'факс.',
    'e.g.', 'i.e.', 'etc.', 'vs.', 'mr.', 'mrs.', 'dr.',
)

_TIMESTAMP_RE = re.compile(r'^(\[\d{1,2}:\d{2}(?::\d{2})?\]\s*)(.*)$')

_UNICODE_TABLE = str.maketrans({
    '\u00a0': ' ',
    '«': '"', '»': '"', '„': '"', '“': '"', '”': '"',
    '‘': "'", '’': "'",
    '…': '...',
    '–': '-', '—': '-',
})
_SPACES_RE = re.compile(r'[ \t]+')
_BEFORE_PUNCT_RE = re.compile(r'\s+([,.;:!?])')
_AFTER_PUNCT_RE = re.compile(r'([,;:!?])(?=[A-Za-zА-Яа-яЁё])')
_SINGLE_DOT_RE = re.compile(r'(?<!\.)\.(?=[A-Za-zА-Яа-яЁё])')
_COMMAS_RE = re.compile(r',+')
_EXCLAMS_RE = re.compile(r'!+')
_QUESTIONS_RE = re.compile(r'\?+')
_DOUBLE_DOT_RE = re.compile(r'(?<!\.)\.\.(?!\.)')
_ANY_SPACE_RE = re.compile(r'\s+')


def _regex_escape(value):
    return re.escape(value)


def _build_replacement_rules():
    rules = []
    for wrong, right in sorted(
            REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(
            r'(?<!\w)' + _regex_escape(wrong) + r'(?!\w)',
            re.IGNORECASE,
        )
        rules.append((pattern, right))
    return rules


_REPLACEMENT_RULES = _build_replacement_rules()

_PROTECT_URL = re.compile(r'(?<!\w)(?:https?://|www\.)\S+', re.IGNORECASE)
_PROTECT_EMAIL = re.compile(
    r'(?<!\w)[\w.+-]+@[\w-]+\.[\w.-]+', re.UNICODE)
_PROTECT_ABBREVIATIONS = [
    re.compile(r'(?<!\w)' + _regex_escape(item) + r'(?!\w)', re.IGNORECASE)
    for item in sorted(_ABBREVIATIONS, key=len, reverse=True)
]

_INTRO_PATTERN = re.compile(
    r'(^|[.!?]\s+)(' +
    '|'.join(_regex_escape(word) for word in
             sorted(INTRO_WORDS, key=len, reverse=True)) +
    r')(\s+)(?=[A-Za-zА-Яа-яЁё])',
    re.IGNORECASE,
)

_DUPLICATE_PATTERN = re.compile(
    r'(?i)(?<!\w)(' + '|'.join(_DUPLICATE_WORDS) +
    r')(\s+)(\1)(?!\w)'
)


class SpeechTextCorrector:
    """Исправляет типичные ошибки распознавания речи.

    ``correct`` можно вызывать многократно: повторная обработка уже
    исправленного текста не должна менять результат.
    """

    def correct(self, text, add_final_period=True):
        if text is None:
            return ''
        text = str(text)
        if not text.strip():
            return text

        result_lines = []
        for line in text.splitlines():
            match = _TIMESTAMP_RE.match(line)
            if match:
                prefix, body = match.group(1), match.group(2)
                result_lines.append(
                    prefix + self._correct_body(body, add_final_period))
            else:
                result_lines.append(
                    self._correct_body(line, add_final_period))

        result = '\n'.join(result_lines)
        if text.endswith(('\n', '\r')):
            result += '\n'
        return result

    # ------------------------------------------------------------------ body
    def _correct_body(self, body, add_final_period):
        if not body or not body.strip():
            return body

        leading = body[:len(body) - len(body.lstrip())]
        trailing = body[len(body.rstrip()):]
        core = body.strip()

        core = self._normalize_unicode(core)
        core, protected = self._protect(core)
        core = self._normalize_punctuation(core)
        core = self._collapse_duplicates(core)
        core = self._apply_replacements(core)
        core = self._add_intro_commas(core)
        core = self._capitalize_sentences(core)
        core = self._restore_protected(core, protected)
        core = self._capitalize_first_cyrillic(core)
        core = self._tidy(core)

        if add_final_period:
            core = self._add_final_period(core)
            core = self._tidy(core)

        return leading + core + trailing

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _normalize_unicode(text):
        return text.translate(_UNICODE_TABLE)

    @staticmethod
    def _protect(text):
        protected = {}
        counter = [0]

        def make_token(original):
            token = f'\ue000P{counter[0]}\ue001'
            counter[0] += 1
            protected[token] = original
            return token

        text = _PROTECT_URL.sub(lambda m: make_token(m.group(0)), text)
        text = _PROTECT_EMAIL.sub(lambda m: make_token(m.group(0)), text)
        for pattern in _PROTECT_ABBREVIATIONS:
            text = pattern.sub(lambda m: make_token(m.group(0)), text)
        return text, protected

    @staticmethod
    def _restore_protected(text, protected):
        for token, original in protected.items():
            text = text.replace(token, original)
        return text

    @staticmethod
    def _normalize_punctuation(text):
        text = _SPACES_RE.sub(' ', text)
        text = _BEFORE_PUNCT_RE.sub(r'\1', text)
        text = _AFTER_PUNCT_RE.sub(r'\1 ', text)
        text = _SINGLE_DOT_RE.sub('. ', text)
        text = _COMMAS_RE.sub(',', text)
        text = _EXCLAMS_RE.sub('!', text)
        text = _QUESTIONS_RE.sub('?', text)
        text = _DOUBLE_DOT_RE.sub('.', text)
        text = _ANY_SPACE_RE.sub(' ', text)
        return text.strip()

    @staticmethod
    def _collapse_duplicates(text):
        while _DUPLICATE_PATTERN.search(text):
            text = _DUPLICATE_PATTERN.sub(lambda m: m.group(1), text)
        return text

    @staticmethod
    def _apply_replacements(text):
        for pattern, replacement in _REPLACEMENT_RULES:
            text = pattern.sub(
                lambda m, rep=replacement: _apply_case(m.group(0), rep),
                text,
            )
        return text

    @staticmethod
    def _add_intro_commas(text):
        def repl(match):
            return match.group(1) + match.group(2) + ', '

        # После замены может появиться ещё одно вводное слово в начале
        # следующего предложения; одного прохода обычно достаточно, но
        # повторяем для устойчивости.
        previous = None
        while previous != text:
            previous = text
            text = _INTRO_PATTERN.sub(repl, text)
        return text

    @staticmethod
    def _capitalize_sentences(text):
        chars = list(text)
        capitalize_next = True
        index = 0
        while index < len(chars):
            char = chars[index]
            if capitalize_next and char.isalpha():
                chars[index] = char.upper()
                capitalize_next = False
            elif char in '.!?\n':
                capitalize_next = True
            index += 1
        return ''.join(chars)

    @staticmethod
    def _capitalize_first_cyrillic(text):
        """Поднимает регистр первой русской буквы (для сокращений и цитат)."""
        for index, char in enumerate(text):
            if char.isalpha():
                if 'А' <= char <= 'я' or char in 'Ёё':
                    return text[:index] + char.upper() + text[index + 1:]
                return text
        return text

    @staticmethod
    def _tidy(text):
        text = _SPACES_RE.sub(' ', text)
        text = _ANY_SPACE_RE.sub(' ', text)
        text = _BEFORE_PUNCT_RE.sub(r'\1', text)
        return text.strip()

    @staticmethod
    def _add_final_period(text):
        text = text.rstrip()
        if not text:
            return text

        text = text.rstrip(' ,;:-')
        if not text:
            return text

        if text[-1] in '.!?…':
            return text

        if text[-1] in '"»)]':
            if len(text) >= 2:
                if text[-2] in ',;:-':
                    return text[:-2] + '.' + text[-1]
                if text[-2] not in '.!?…':
                    return text[:-1] + '.' + text[-1]
            return text

        return text + '.'


def _apply_case(original, replacement):
    if not replacement:
        return replacement
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


_DEFAULT_CORRECTOR = SpeechTextCorrector()


@lru_cache(maxsize=2048)
def _correct_cached(text, add_final_period):
    return _DEFAULT_CORRECTOR.correct(text, add_final_period=add_final_period)


def clear_text_correction_cache():
    _correct_cached.cache_clear()


def correct_speech_text(text, enabled=True, add_final_period=True):
    """Возвращает исправленный текст или исходный, если ``enabled=False``."""
    if not enabled:
        return '' if text is None else text
    if text is None:
        return ''
    return _correct_cached(str(text), bool(add_final_period))
