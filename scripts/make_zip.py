"""Empacota somente os arquivos públicos selecionados para entrega."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = (
    ".env.example",
    ".gitignore",
    "README.md",
    "app.py",
    "pytest.ini",
    "requirements.txt",
    "requirements-dev.txt",
    "data/clientes.csv",
    "data/score_limite.csv",
    "data/solicitacoes_aumento_limite.csv",
    "docs/ARQUITETURA.md",
    "docs/OPERACAO.md",
    "docs/TESTES.md",
    "scripts/make_zip.sh",
    "scripts/make_zip.py",
    "scripts/test_local.py",
)


def public_files(root: Path) -> list[Path]:
    """Inclui os arquivos fixos e os módulos Python da aplicação e dos testes."""
    names = set(PUBLIC_FILES)
    for directory in ("src/banco_agil", "tests"):
        for path in (root / directory).rglob("*.py"):
            relative = path.relative_to(root)
            if not any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                names.add(relative.as_posix())
    files = []
    for name in sorted(names):
        path = root / name
        # Recusa links para não incluir arquivos de fora da seleção.
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent):
            raise ValueError(f"Link simbólico não permitido na entrega: {name}")
        if not path.is_file():
            raise ValueError(f"Arquivo da entrega não encontrado: {name}")
        files.append(path)
    return files


def create_archive(root: Path, destination: Path) -> list[str]:
    files = public_files(root)
    names = [path.relative_to(root).as_posix() for path in files]
    destination = destination.resolve()
    if destination.suffix.lower() != ".zip":
        raise ValueError("O destino precisa ter a extensão .zip.")
    if destination in [path.resolve() for path in files]:
        raise ValueError("O destino não pode substituir um arquivo da entrega.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, name in zip(files, names):
                archive.write(path, name)
        with zipfile.ZipFile(temporary) as archive:
            if archive.namelist() != names or archive.testzip() is not None:
                raise ValueError("Não foi possível validar o pacote gerado.")
        # Só substitui um pacote anterior depois que o novo estiver completo.
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path, default=ROOT / "desafio.zip",
                        help="Caminho do ZIP; o padrão é desafio.zip na raiz do projeto.")
    parser.add_argument("--list", action="store_true", help="Mostra os arquivos sem criar o ZIP.")
    args = parser.parse_args()
    try:
        if args.list:
            for path in public_files(ROOT):
                print(path.relative_to(ROOT).as_posix())
        else:
            names = create_archive(ROOT, args.destination)
            print(f"Pacote criado com {len(names)} arquivos: {args.destination.resolve()}")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Falha ao preparar a entrega: {error}\n")


if __name__ == "__main__":
    main()
