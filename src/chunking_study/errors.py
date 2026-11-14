"""Erros de configuração e de provedor.

Mensagem não inclui chave de API. Quem chama decide se vira saída 1.
"""


class ConfigError(ValueError):
    """Variável de ambiente inválida ou SDK pedido sem estar instalado."""


class ProviderError(RuntimeError):
    """O provedor devolveu um embedding que o índice não consegue usar."""
