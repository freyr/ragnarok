import pytest

from ragnarok.redact import MARK, redact

# Synthetic test values, built at runtime so no literal secret sits in the repo.
AWS = "AKIA" + "ABCDEFGHIJKLMNOP"
GH = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2"
SK = "sk-" + "proj" + "A1b2C3d4E5f6G7h8I9j0"
SLACK = "xoxb-" + "1234567890-abcdefghij"
KEY_BLOCK = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"


@pytest.mark.parametrize("secret", [AWS, GH, SK, SLACK, KEY_BLOCK])
def test_tokens_are_redacted(secret):
    text, n = redact(f"before {secret} after")
    assert secret not in text and MARK in text and n == 1


@pytest.mark.parametrize(
    "line, kept",
    [
        ("password: Hunter2Hunter2", "password: "),
        ("DB_PASSWORD=s3cr3tValue", "DB_PASSWORD="),
        ('api_key = "abcd1234efgh"', 'api_key = "'),
        ("client_secret: 'zzzz9999'", "client_secret: '"),
        ("RABBITMQ_DEFAULT_PASS: guest1", "RABBITMQ_DEFAULT_PASS: "),
        ("--password=rootpw123 \\", "--password="),
    ],
)
def test_key_value_literals_are_redacted(line, kept):
    text, n = redact(line)
    assert text.startswith(kept) and MARK in text and n == 1


@pytest.mark.parametrize(
    "line",
    [
        "password: !secret wifi_password",
        "password: ${DB_PASSWORD}",
        "token: <your-token>",
        "password: {{ vault_password }}",
        "task-by-task review of the plan",
        "The password policy is strict.",
        'data-token-slot="php-new"',
        "max_tokens: 4096",
        "Second pass: refine to 2 arc minutes",
        "bool $passive = false,",
        "token: string|null,",
        "password: getenv('AMQP_PASSWORD') ?: 'guest',",
    ],
)
def test_references_and_prose_stay(line):
    assert redact(line) == (line, 0)


def test_url_credentials():
    text, n = redact("mysql://root:Pa55word@db:3306/app")
    assert text == f"mysql://root:{MARK}@db:3306/app" and n == 1
