"""Executa o pytest com conexões Python restritas ao loopback local."""
from __future__ import annotations

import ipaddress
import os
from pathlib import Path
import socket
import sys


def require_loopback(host) -> None:
    if isinstance(host, bytes):
        host = host.decode("ascii")
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise RuntimeError("Conexão externa bloqueada durante os testes locais.")


def guard_connection(original):
    def guarded(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            require_loopback(address[0])
        return original(sock, address)
    return guarded


def guard_resolution(original):
    def guarded(host, *args, **kwargs):
        if host is not None:
            require_loopback(host)
        return original(host, *args, **kwargs)
    return guarded


def guard_datagram(original):
    def guarded(sock, *args, **kwargs):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            require_loopback(args[-1][0])
        return original(sock, *args, **kwargs)
    return guarded


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    socket.socket.connect = guard_connection(socket.socket.connect)
    socket.socket.connect_ex = guard_connection(socket.socket.connect_ex)
    socket.socket.sendto = guard_datagram(socket.socket.sendto)
    socket.getaddrinfo = guard_resolution(socket.getaddrinfo)

    # Aplica a proteção antes de carregar o pytest e seus plugins.
    import pytest

    return pytest.main(["-q", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
