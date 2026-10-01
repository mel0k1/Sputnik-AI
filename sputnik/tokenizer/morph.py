import re

BOW = "\u2581"
MID = "\u00b7"

PREFIXES = (
    "электро", "авиа", "авто", "агро", "анти", "архи", "астро", "аудио",
    "гидро", "гипер", "гипо", "квази", "интер", "инфра", "мега", "микро",
    "мини", "моно", "мото", "нано", "нейро", "пост", "псевдо",
    "радио", "рентген", "сверх", "супер", "теле", "транс", "ультра", "фото",
    "эко", "между", "после", "взаим", "само", "вне",
    "пере", "пре", "при", "про", "под", "над", "раз", "рас", "воз", "вос",
    "без", "бес", "пред", "до", "вы", "за", "на", "не", "ни", "об", "от",
    "со", "из", "ис", "по",
)
ENDINGS = (
    "ироваться", "ированием", "ированию", "ирования", "ирование",
    "ируется", "ируются", "ировать", "ировал", "изация", "изации",
    "оваться", "овано", "уется", "уются", "овать",
    "аться", "еться", "яться", "ться", "тся", "ся",
    "ствами", "ствам", "ствах", "ство", "ства",
    "ациями", "ациях", "ацию", "ации", "ация",
    "ениями", "ениям", "ением", "ения", "ение",
    "аниями", "аниям", "анием", "ания", "ание",
    "остей", "остям", "остях", "остью", "ость", "ости",
    "изм", "измы", "изма", "изме", "ист", "исты", "иста",
    "ешься", "ишься", "ется", "ются", "утся", "атся", "ятся",
    "ешь", "ишь", "ует", "ают", "яют", "ет", "ют", "ат", "ят", "ит",
    "шему", "шему", "ому", "ему", "ыми", "ими", "ого", "его",
    "ую", "юю", "ой", "ей", "ым", "им", "ья", "ье",
    "ах", "ях", "ов", "ев", "ам", "ям", "ый", "ий", "ая", "яя", "ое", "ее",
    "ы", "и", "а", "я", "е", "у", "ю", "ь", "й", "о",
)

PREFIXES = tuple(sorted(set(p.strip() for p in PREFIXES), key=len, reverse=True))
ENDINGS = tuple(sorted(set(ENDINGS), key=len, reverse=True))

TOKEN_RE = re.compile(r"\w+(?:-\w+)*|[^\w\s]+", re.UNICODE)


def split_morphs(word, depth=0):
    if len(word) < 5 or depth >= 2:
        return [word]
    for p in PREFIXES:
        if word.startswith(p) and len(word) - len(p) >= 4:
            return [p] + split_morphs(word[len(p):], depth + 1)
    for e in ENDINGS:
        min_stem = 6 if len(e) < 2 else 4
        if word.endswith(e) and len(word) - len(e) >= min_stem:
            return split_morphs(word[:-len(e)], depth + 1) + [e]
    return [word]


def _digits(tok):
    groups = []
    i = len(tok) % 3
    if i:
        groups.append(tok[:i])
    while i < len(tok):
        groups.append(tok[i:i + 3])
        i += 3
    return groups


def pretokenize(text):
    text = text.lower().replace("ё", "е").replace(BOW, "_").replace(MID, ".")
    out = []
    last = 0
    for m in TOKEN_RE.finditer(text):
        if "\n" in text[last:m.start()]:
            out.append("\n")
        tok = m.group()
        if not (tok[0].isalnum() or tok[0] == "_"):
            out.append(tok)  # пунктуация без маркера — клеится к предыдущему слову
        elif tok[0].isdigit():
            gs = _digits(tok)
            out.append(BOW + gs[0])
            out.extend(MID + g for g in gs[1:])
        elif "-" in tok:
            parts = tok.split("-")
            cs = split_morphs(parts[0])
            out.append(BOW + cs[0])
            out.extend(MID + c for c in cs[1:])
            for p in parts[1:]:
                if not p:
                    continue
                cs = split_morphs(p)
                out.append(MID + "-" + cs[0])
                out.extend(MID + c for c in cs[1:])
        else:
            cs = split_morphs(tok)
            out.append(BOW + cs[0])
            out.extend(MID + c for c in cs[1:])
        last = m.end()
    return out
