"""One module per stylized fact.

Each exposes ``FACT``, ``FACT_ID``, ``PRIORITY``, ``TITLE`` and a ``run(ctx)``
that returns a summary dict.  The driver discovers them through
``config.ENABLED_FACTS``.
"""
