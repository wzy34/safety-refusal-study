"""Versioned prompt templates in `prompts/`. A template's version is a hash of its text, and
that version is recorded with every output produced from it."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from srs.ids import sha256_hex

_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class TemplateError(Exception):
    pass


@dataclass(frozen=True)
class Template:
    name: str
    text: str

    @property
    def version(self) -> str:
        return sha256_hex(self.text)[:12]

    @property
    def variables(self) -> set[str]:
        return set(_VAR.findall(self.text))

    def render(self, **values: str) -> str:
        missing = self.variables - set(values)
        if missing:
            raise TemplateError(f"{self.name}: missing template variables {sorted(missing)}")
        return _VAR.sub(lambda m: str(values[m.group(1)]), self.text)


def render_string(text: str, **values: str) -> str:
    return Template("<inline>", text).render(**values)


def load_template(prompts_dir: Path, name: str) -> Template:
    path = prompts_dir / name
    if not path.exists():
        raise TemplateError(f"template not found: {path}")
    return Template(name, path.read_text(encoding="utf-8"))
