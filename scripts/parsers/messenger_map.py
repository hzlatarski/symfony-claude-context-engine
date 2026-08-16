"""Symfony Messenger contract parser.

Links a message class to the handler(s) that consume it and the call sites
that dispatch it — the async equivalent of the ``fetch()`` → route edges the
call graph already resolves for the JS ↔ PHP boundary. Answers:

    "Which handler runs this message?"
    "Who dispatches this message?"
    "Is anything dispatched but never handled?" (orphan producer)
    "Is a handler wired up but never dispatched?" (orphan consumer)

Detection (regex, mirroring ``route_map`` — no tree-sitter dependency, works
even where the call-graph parser can't load). Comments are blanked first
(newline-preserving) so disabled handlers and dispatch-in-docblocks don't
register:

- **Handlers** — a class carrying ``#[AsMessageHandler]`` (class-level, the
  Symfony 7 norm) consumes the type of the first parameter of its ``__invoke``.
  Method-level ``#[AsMessageHandler]`` (on a named method), an explicit
  ``handles:``/positional ``X::class`` argument, a ``method:`` argument, and
  union-typed first params are all supported.
- **Producers** — ``->dispatch(new X(...))`` and
  ``->dispatchAfterCurrentBus(new X(...))`` call sites, unwrapping a
  ``new Envelope(new X(...))`` wrapper to the inner message.

Message identity is the resolved FQCN (via the file's ``use`` imports —
including grouped ``use A\B\{X, Y};`` and relative-qualified names), so a
handler and a producer in different files associate correctly.

Known limits (documented, not silent):
- Only messenger messages under the ``App\\Message\\`` namespace — or a class
  that actually has a handler — are tracked as producers. This keeps Symfony
  EventDispatcher ``->dispatch(new App\\Event\\...)`` calls out of the orphan
  list; the app dispatches domain events through the same method name.
- ``->dispatch($variable)`` (non-literal) is not resolved. Those sites are
  counted in ``stats.unresolved_dispatch`` (which may include EventDispatcher
  variable dispatch) so the blind spot is visible, and a handler with no
  *literal* producer is reported as ``undispatched`` rather than "dead".
- Producers are scanned under ``src/`` only; a message dispatched solely from
  ``config/`` or fixtures will read as ``undispatched``.
"""
from __future__ import annotations

import re
from pathlib import Path

_MESSAGE_NAMESPACE = "App\\Message\\"

# namespace App\MessageHandler;
_NS_RE = re.compile(r"^\s*namespace\s+([\w\\]+)\s*;", re.MULTILINE)

# use App\Message\FooMessage;  /  use App\Message\FooMessage as Bar;
_USE_RE = re.compile(r"^\s*use\s+([\w\\]+)(?:\s+as\s+(\w+))?\s*;", re.MULTILINE)
# use App\Message\{FooMessage, BarMessage as Baz};
_USE_GROUP_RE = re.compile(r"^\s*use\s+([\w\\]+)\\\{([^}]+)\}\s*;", re.MULTILINE)

# The first type declaration in a file — marks the end of the import header.
_TYPE_DECL_RE = re.compile(r"(?:^|\s)(?:final\s+|abstract\s+|readonly\s+)*\b(?:class|interface|trait|enum)\s+\w+")

# A class declaration (preceded by start/whitespace so ``Foo::class`` is excluded).
_CLASS_DECL_RE = re.compile(r"(?:^|\s)(?:final\s+|abstract\s+|readonly\s+)*class\s+(\w+)")

# #[AsMessageHandler]  or  #[AsMessageHandler(bus: 'x')]  — terminates on ] or ,
# (so a grouped attribute list ``#[AsMessageHandler, Other]`` still matches).
_ATTR_RE = re.compile(r"#\[\s*AsMessageHandler\b(?:\s*\((?P<args>.*?)\))?\s*(?=[,\]])", re.DOTALL)

# Head of a visibility-qualified method: captures name, leaves cursor at '('.
_METHOD_HEAD_RE = re.compile(r"(?:public|protected|private)\s+function\s+(\w+)\s*\(")
# Head of __invoke, leaves cursor at '('.
_INVOKE_HEAD_RE = re.compile(r"function\s+__invoke\s*\(")

# First typed parameter type token(s): `?App\Foo|Bar $name` -> `App\Foo|Bar`.
_FIRST_PARAM_TYPE_RE = re.compile(r"^\s*\??\s*([\\\w|&]+)\s+(?:&\s*|\.\.\.\s*)?\$\w+")

# handles: Foo::class  /  positional Foo::class  /  method: 'handleIt'
_HANDLES_NAMED_RE = re.compile(r"handles\s*:\s*([\\\w]+)::class")
_HANDLES_POS_RE = re.compile(r"^\s*([\\\w]+)::class")
_METHOD_ARG_RE = re.compile(r"method\s*:\s*['\"](\w+)['\"]")

# ->dispatch(new Foo(  /  ->dispatchAfterCurrentBus(new Foo(
_DISPATCH_NEW_RE = re.compile(
    r"->\s*(?P<via>dispatch|dispatchAfterCurrentBus)\s*\(\s*new\s+(?P<type>[\\\w]+)"
)
# ->dispatch($var  — a non-literal (variable) dispatch, unresolved.
_DISPATCH_VAR_RE = re.compile(r"->\s*(?:dispatch|dispatchAfterCurrentBus)\s*\(\s*\$")
# new Inner(  inside an Envelope wrapper.
_ENVELOPE_INNER_RE = re.compile(r"new\s+([\\\w]+)")


def _blank_comments(content: str) -> str:
    """Replace PHP comment bodies with spaces, preserving newlines (and offsets).

    Blanks ``/* ... */``, ``// ...`` and ``# ...`` (but NOT ``#[`` attributes).
    Keeping length + newlines intact means ``str.count("\\n", 0, pos)`` line
    numbers stay correct after blanking.
    """
    def _blank(m: re.Match) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))

    content = re.sub(r"/\*.*?\*/", _blank, content, flags=re.DOTALL)
    # Line comments: `//` and `#` (not `#[`). Guard `://` so URLs in strings survive.
    content = re.sub(r"(?<!:)//[^\n]*", _blank, content)
    content = re.sub(r"#(?!\[)[^\n]*", _blank, content)
    return content


def _line_of(content: str, pos: int) -> int:
    """1-based line number of ``pos`` in ``content``."""
    return content.count("\n", 0, pos) + 1


def _fqcn(namespace: str, class_name: str) -> str:
    return f"{namespace}\\{class_name}" if namespace and class_name else class_name


def _build_imports(content: str) -> dict[str, str]:
    """Map short name (alias or last segment) -> FQCN from ``use`` statements.

    Scoped to the import header (before the first type declaration) so trait
    ``use`` inside a class body and stray ``use`` lines inside heredocs don't
    pollute the map. Handles grouped ``use A\\B\\{X, Y as Z};``.
    """
    header_end = None
    decl = _TYPE_DECL_RE.search(content)
    if decl:
        header_end = decl.start()
    header = content[:header_end] if header_end is not None else content

    imports: dict[str, str] = {}
    for m in _USE_GROUP_RE.finditer(header):
        base = m.group(1)
        for raw in m.group(2).split(","):
            member = raw.strip()
            if not member:
                continue
            alias = None
            am = re.match(r"([\w\\]+)\s+as\s+(\w+)", member)
            if am:
                member, alias = am.group(1), am.group(2)
            short = alias if alias else member.split("\\")[-1]
            imports[short] = f"{base}\\{member}"
    for m in _USE_RE.finditer(header):
        fqcn = m.group(1)
        if fqcn in ("function", "const"):  # `use function ...` / `use const ...`
            continue
        alias = m.group(2)
        short = alias if alias else fqcn.split("\\")[-1]
        imports[short] = fqcn
    return imports


def _resolve(type_name: str, imports: dict[str, str], current_ns: str) -> str:
    """Resolve a written type name to a FQCN using the file's imports.

    - leading ``\\`` → already fully-qualified.
    - relative ``A\\B`` → resolve the first segment via imports, else prepend ns.
    - bare ``A`` → imports, else current namespace.
    """
    if not type_name:
        return ""
    if type_name.startswith("\\"):
        return type_name.lstrip("\\")
    if "\\" in type_name:
        head, _, rest = type_name.partition("\\")
        if head in imports:
            return f"{imports[head]}\\{rest}"
        return f"{current_ns}\\{type_name}" if current_ns else type_name
    if type_name in imports:
        return imports[type_name]
    return f"{current_ns}\\{type_name}" if current_ns else type_name


def _slice_params(text: str, open_paren_idx: int) -> str:
    """Inner text of the parenthesis group starting at ``open_paren_idx`` ('(').

    Depth-matched so a parameter attribute (``#[Foo(bar)]``) or a default value
    containing ``)`` doesn't truncate the parameter list early.
    """
    depth = 0
    for i in range(open_paren_idx, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_idx + 1:i]
    return text[open_paren_idx + 1:]


def _param_message_types(params: str) -> list[str]:
    """Type name(s) of the first parameter, splitting a union into members."""
    # Strip any leading parameter attributes (#[...]) — possibly several.
    stripped = params
    while True:
        new = re.sub(r"^\s*#\[.*?\]\s*", "", stripped, flags=re.DOTALL)
        if new == stripped:
            break
        stripped = new
    first = stripped.split(",", 1)[0]
    m = _FIRST_PARAM_TYPE_RE.match(first)
    if not m:
        return []
    raw = m.group(1)
    parts = [p.strip() for p in re.split(r"[|&]", raw) if p.strip()]
    return parts


def _message_from_attr_args(args: str, imports: dict[str, str], current_ns: str) -> str:
    """FQCN from an explicit ``handles: Foo::class`` / positional ``Foo::class``."""
    if not args:
        return ""
    named = _HANDLES_NAMED_RE.search(args)
    if named:
        return _resolve(named.group(1), imports, current_ns)
    pos = _HANDLES_POS_RE.match(args)
    if pos:
        return _resolve(pos.group(1), imports, current_ns)
    return ""


def _enclosing_class(content: str, pos: int, current_ns: str) -> str:
    """FQCN of the class whose declaration most closely precedes ``pos``."""
    last = None
    for m in _CLASS_DECL_RE.finditer(content, 0, pos):
        last = m
    return _fqcn(current_ns, last.group(1)) if last else ""


def _parse_content(content: str, rel_path: str) -> tuple[list[dict], list[dict]]:
    """Parse one PHP file's text into (handlers, producers).

    handlers:  {message_fqcn, message_short, handler_class, handler_method, file, line, confidence}
    producers: {message_fqcn, message_short, file, line, via, confidence}
    """
    blanked = _blank_comments(content)

    ns_match = _NS_RE.search(blanked)
    current_ns = ns_match.group(1) if ns_match else ""
    imports = _build_imports(blanked)

    handlers: list[dict] = []
    producers: list[dict] = []

    # --- Handlers -----------------------------------------------------------
    if "AsMessageHandler" in blanked:
        for attr in _ATTR_RE.finditer(blanked):
            args = (attr.group("args") or "").strip()
            tail = blanked[attr.end():]

            cls = _CLASS_DECL_RE.search(tail)
            mh = _METHOD_HEAD_RE.search(tail)
            method_level = mh is not None and (cls is None or mh.start() < cls.start())

            if method_level:
                handler_class = _enclosing_class(blanked, attr.start(), current_ns)
                method = mh.group(1)
                params = _slice_params(tail, mh.end() - 1)
            else:
                if cls is None:
                    continue
                handler_class = _fqcn(current_ns, cls.group(1))
                method = "__invoke"
                # Bound the __invoke search to this class body (up to the next class).
                body = tail[cls.end():]
                nxt = _CLASS_DECL_RE.search(body)
                if nxt is not None:
                    body = body[:nxt.start()]
                ih = _INVOKE_HEAD_RE.search(body)
                params = _slice_params(body, ih.end() - 1) if ih is not None else ""

            method_override = _METHOD_ARG_RE.search(args)
            if method_override:
                method = method_override.group(1)

            explicit = _message_from_attr_args(args, imports, current_ns)
            if explicit:
                message_fqcns = [explicit]
            else:
                message_fqcns = [
                    _resolve(t, imports, current_ns) for t in _param_message_types(params)
                ]
            for message_fqcn in message_fqcns:
                if not message_fqcn:
                    continue
                handlers.append({
                    "message_fqcn": message_fqcn,
                    "message_short": message_fqcn.split("\\")[-1],
                    "handler_class": handler_class,
                    "handler_method": method,
                    "file": rel_path,
                    "line": _line_of(blanked, attr.start()),
                    "confidence": 1.0,
                })

    # --- Producers ----------------------------------------------------------
    for m in _DISPATCH_NEW_RE.finditer(blanked):
        type_name = m.group("type")
        # Unwrap ->dispatch(new Envelope(new Inner(...))) to the inner message.
        if type_name.split("\\")[-1] == "Envelope":
            inner = _ENVELOPE_INNER_RE.search(blanked[m.end():])
            if inner is None:
                continue
            type_name = inner.group(1)
        fqcn = _resolve(type_name, imports, current_ns)
        if not fqcn:
            continue
        producers.append({
            "message_fqcn": fqcn,
            "message_short": fqcn.split("\\")[-1],
            "file": rel_path,
            "line": _line_of(blanked, m.start()),
            "via": m.group("via"),
            "confidence": 1.0,
        })

    return handlers, producers


def parse(project_root: Path) -> dict:
    """Build the messenger contract map across ``src/**/*.php``.

    Returns:
        {
          "messages": {
             fqcn: {"short", "namespace", "handlers": [...], "producers": [...]},
             ...
          },
          "orphans": {"unhandled": [fqcn, ...], "undispatched": [fqcn, ...]},
          "stats": {...},
        }
    """
    src_dir = project_root / "src"
    php_files = sorted(src_dir.rglob("*.php")) if src_dir.is_dir() else []

    all_handlers: list[dict] = []
    all_producers: list[dict] = []
    unresolved_dispatch = 0

    for path in php_files:
        rel_path = path.relative_to(project_root).as_posix()
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "AsMessageHandler" not in content and "dispatch" not in content:
            continue
        handlers, producers = _parse_content(content, rel_path)
        all_handlers.extend(handlers)
        all_producers.extend(producers)
        # Count variable (non-literal) dispatches so the resolver's blind spot
        # is visible. Uses the dispatch regex directly (no window heuristic), so
        # a deeply-indented literal `new X()` is never double-counted.
        unresolved_dispatch += len(_DISPATCH_VAR_RE.findall(_blank_comments(content)))

    handled_fqcns = {h["message_fqcn"] for h in all_handlers}

    messages: dict[str, dict] = {}

    def _entry(fqcn: str) -> dict:
        e = messages.get(fqcn)
        if e is None:
            e = {
                "short": fqcn.split("\\")[-1],
                "namespace": fqcn.rsplit("\\", 1)[0] if "\\" in fqcn else "",
                "handlers": [],
                "producers": [],
            }
            messages[fqcn] = e
        return e

    for h in all_handlers:
        _entry(h["message_fqcn"])["handlers"].append(h)

    for p in all_producers:
        fqcn = p["message_fqcn"]
        # Only associate producers that are genuinely messenger messages:
        # under App\Message\, or a class that actually has a handler. This
        # excludes EventDispatcher ->dispatch(new App\Event\...) calls.
        if fqcn.startswith(_MESSAGE_NAMESPACE) or fqcn in handled_fqcns:
            _entry(fqcn)["producers"].append(p)

    unhandled = sorted(
        fqcn for fqcn, e in messages.items() if not e["handlers"] and e["producers"]
    )
    undispatched = sorted(
        fqcn for fqcn, e in messages.items() if e["handlers"] and not e["producers"]
    )
    handled = [fqcn for fqcn, e in messages.items() if e["handlers"]]

    return {
        "messages": messages,
        "orphans": {
            "unhandled": unhandled,
            "undispatched": undispatched,
        },
        "stats": {
            "total_messages": len(handled),
            "total_handlers": len(all_handlers),
            "total_producer_sites": sum(len(e["producers"]) for e in messages.values()),
            "unhandled": len(unhandled),
            "undispatched": len(undispatched),
            "unresolved_dispatch": unresolved_dispatch,
        },
    }


def summary(project_root: Path) -> str:
    """Short one-line summary for session-start context."""
    result = parse(project_root)
    s = result["stats"]
    return (
        f"{s['total_messages']} messages, {s['total_handlers']} handlers, "
        f"{s['total_producer_sites']} producer sites "
        f"({s['unhandled']} unhandled, {s['undispatched']} undispatched)"
    )
