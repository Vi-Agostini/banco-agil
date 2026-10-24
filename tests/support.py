"""Verificações compartilhadas dos dados usados nos testes."""


def snapshot_csv(repo):
    """Lê os CSVs da fixture sem alterar seu conteúdo."""
    return {p.name: p.read_bytes() for p in repo.dir.glob("*.csv")}
