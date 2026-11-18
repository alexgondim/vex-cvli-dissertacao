"""Leitor de planilha .xlsx com a biblioteca padrão (zipfile e ElementTree).

Cobre o que a planilha de origem usa: textos compartilhados (sharedStrings), texto em linha (inlineStr), número,
booleano e data serial, reconhecida pelo formato numérico da célula, e não pelo nome da coluna. `ler_tabela` devolve
o cabeçalho, as linhas e os nomes das abas. Não grava arquivo.

É importado pelo `gerar_base_bairro_mes.py`.
"""
import re
import zipfile
import datetime as _dt
import xml.etree.ElementTree as ET

_NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_NS_REL_DOC = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_NS_REL_PKG = "{http://schemas.openxmlformats.org/package/2006/relationships}"

# numFmtId embutidos que representam data/hora (ECMA-376, parte 1, 18.8.30).
_BUILTIN_DATE_FMTS = set(range(14, 23)) | {27, 30, 36, 45, 46, 47, 50, 57}
_DATE_CODE_RE = re.compile(r"(\[\$-[^\]]*\])|(\"[^\"]*\")")


def _col_letters(ref):
    out = []
    for ch in ref:
        if ch.isalpha():
            out.append(ch)
        else:
            break
    return "".join(out)


def _col_index(ref):
    n = 0
    for ch in _col_letters(ref):
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def _is_date_fmt_code(code):
    """Sobra do codigo de formato apos tirar literais e locale: se ainda tem
    y/m/d/h/s fora de escape, e data."""
    limpo = _DATE_CODE_RE.sub("", code)
    limpo = re.sub(r"\.", "", limpo)
    return bool(re.search(r"[ymdhs]", limpo, re.IGNORECASE))


class Xlsx:
    def __init__(self, path):
        self.path = str(path)
        self._zip = zipfile.ZipFile(self.path)
        self._shared = self._load_shared_strings()
        self._date_xf = self._load_date_styles()
        self.sheets = self._load_sheet_index()

    def close(self):
        self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _read(self, name):
        try:
            return self._zip.read(name)
        except KeyError:
            return None

    def _load_shared_strings(self):
        raw = self._read("xl/sharedStrings.xml")
        if raw is None:
            return []
        root = ET.fromstring(raw)
        out = []
        for si in root.findall(f"{_NS_MAIN}si"):
            # Texto pode vir partido em varios runs (<r><t>) por formatacao.
            partes = [t.text or "" for t in si.iter(f"{_NS_MAIN}t")]
            out.append("".join(partes))
        return out

    def _load_date_styles(self):
        raw = self._read("xl/styles.xml")
        if raw is None:
            return set()
        root = ET.fromstring(raw)
        custom = {}
        for nf in root.iter(f"{_NS_MAIN}numFmt"):
            fid = int(nf.get("numFmtId"))
            custom[fid] = nf.get("formatCode") or ""
        datas = set()
        cell_xfs = root.find(f"{_NS_MAIN}cellXfs")
        if cell_xfs is None:
            return datas
        for i, xf in enumerate(cell_xfs.findall(f"{_NS_MAIN}xf")):
            fid = int(xf.get("numFmtId") or 0)
            if fid in _BUILTIN_DATE_FMTS:
                datas.add(i)
            elif fid in custom and _is_date_fmt_code(custom[fid]):
                datas.add(i)
        return datas

    def _load_sheet_index(self):
        rels_raw = self._read("xl/_rels/workbook.xml.rels")
        rid_para_alvo = {}
        if rels_raw is not None:
            for rel in ET.fromstring(rels_raw).findall(f"{_NS_REL_PKG}Relationship"):
                alvo = rel.get("Target")
                if alvo.startswith("/"):
                    alvo = alvo[1:]
                elif not alvo.startswith("xl/"):
                    alvo = "xl/" + alvo
                rid_para_alvo[rel.get("Id")] = alvo
        wb = ET.fromstring(self._read("xl/workbook.xml"))
        sheets = []
        for sh in wb.iter(f"{_NS_MAIN}sheet"):
            rid = sh.get(f"{_NS_REL_DOC}id")
            sheets.append({"name": sh.get("name"), "path": rid_para_alvo.get(rid)})
        return sheets

    def _serial_para_data(self, valor):
        # Epoca 1900 com o bug do ano bissexto de 1900 herdado do Lotus.
        if valor is None:
            return None
        dias = int(valor)
        frac = float(valor) - dias
        if dias >= 60:
            dias -= 1
        base = _dt.datetime(1899, 12, 31) + _dt.timedelta(days=dias)
        return base + _dt.timedelta(seconds=round(frac * 86400))

    def rows(self, sheet=0):
        """Gera cada linha como lista de valores ja convertidos, preenchendo
        buracos de celula (<c> ausente) com None para nao desalinhar coluna."""
        alvo = self.sheets[sheet]["path"] if isinstance(sheet, int) else None
        if alvo is None:
            for s in self.sheets:
                if s["name"] == sheet:
                    alvo = s["path"]
                    break
        raw = self._read(alvo)
        root = ET.fromstring(raw)
        for row in root.iter(f"{_NS_MAIN}row"):
            celulas = {}
            maxi = -1
            for c in row.findall(f"{_NS_MAIN}c"):
                ref = c.get("r") or ""
                idx = _col_index(ref) if _col_letters(ref) else (maxi + 1)
                maxi = max(maxi, idx)
                celulas[idx] = self._valor_celula(c)
            yield [celulas.get(i) for i in range(maxi + 1)]

    def _valor_celula(self, c):
        t = c.get("t")
        if t == "inlineStr":
            is_el = c.find(f"{_NS_MAIN}is")
            if is_el is None:
                return None
            return "".join(x.text or "" for x in is_el.iter(f"{_NS_MAIN}t"))
        v = c.find(f"{_NS_MAIN}v")
        if v is None or v.text is None:
            return None
        bruto = v.text
        if t == "s":
            i = int(bruto)
            return self._shared[i] if i < len(self._shared) else None
        if t == "b":
            return bruto == "1"
        if t in ("str", "e"):
            return bruto
        try:
            num = float(bruto)
        except ValueError:
            return bruto
        s_attr = c.get("s")
        if s_attr is not None and int(s_attr) in self._date_xf:
            return self._serial_para_data(num)
        return int(num) if num.is_integer() else num


def ler_tabela(path, sheet=0, linha_cabecalho=0):
    """Devolve (cabecalho, linhas). Linhas sao normalizadas ao tamanho do
    cabecalho — planilha costuma ter linha curta no fim."""
    with Xlsx(path) as wb:
        todas = list(wb.rows(sheet))
        nomes_abas = [s["name"] for s in wb.sheets]
    if not todas:
        return [], [], nomes_abas
    cab = [("" if x is None else str(x).strip()) for x in todas[linha_cabecalho]]
    n = len(cab)
    corpo = []
    for r in todas[linha_cabecalho + 1:]:
        if all(x is None or (isinstance(x, str) and not x.strip()) for x in r):
            continue
        r = list(r[:n]) + [None] * max(0, n - len(r))
        corpo.append(r)
    return cab, corpo, nomes_abas
