garak.generators.rest
=====================

Flexible connector for REST-based APIs.

Uses the following options from ``_config.plugins.generators["rest.RestGenerator"]``:

* ``uri`` - (optional) the URI of the REST endpoint; this can also be passed in --target_name
* ``name`` - a short name for this service; defaults to the uri
* ``key_env_var`` - (optional) the name of the environment variable holding an API key, by default ``REST_API_KEY``
* ``req_template`` - a string where the text ``$KEY`` is replaced by env var ``REST_API_KEY``'s value (or whatever's specified in ``key_env_var``) and ``$INPUT`` is replaced by the prompt. Default is to just send the input text.
* ``req_template_json_object`` - (optional) the request template as a Python object, to be serialised as a JSON string before replacements
* ``method`` - a string describing the HTTP method, to be passed to the requests module; default "post".
* ``headers`` - dict describing HTTP headers to be sent with the request
* ``proxies`` - dict passed to ``requests`` method call. See `required format <https://requests.readthedocs.io/en/latest/user/advanced/#proxies>`_.
* ``response_json`` - Is the response in JSON format? (bool)
* ``response_json_field`` - (optional) Which field of the response JSON should be used as the output string? Default ``text``. Can also be a JSONPath value, and ``response_json_field`` is used as such if it starts with ``$``.
* ``request_timeout`` - How many seconds should we wait before timing out? Default 20
* ``ratelimit_codes`` - Which endpoint HTTP response codes should be caught as indicative of rate limiting and retried? ``List[int]``, default ``[429]``
* ``skip_codes`` - Which endpoint HTTP response code should lead to the generation being treated as not possible and skipped for this query. Takes precedence over ``ratelimit_codes``.
* ``verify_ssl`` - (optional) Enforce ssl certificate validation? Default is ``True``, a file path to a CA bundle can be provided. (bool|str)
* ``client_cert`` - (optional) Path to a PEM-encoded client certificate file for mTLS. If the file contains both the certificate and private key, ``client_key`` is not required.
* ``client_key`` - (optional) Path to the client private key file for mTLS. Must be set if ``client_cert`` does not contain the key. Cannot be set without ``client_cert``.
* ``client_key_passphrase_env_var`` - (optional) Name of an environment variable holding the passphrase for an encrypted ``client_key``. The passphrase is never read from config files. If set, the named env var must be defined or startup will fail.
* ``response_json_join`` - (optional) When the JSONPath in ``response_json_field`` finds several values (e.g. a reply split into text parts and button labels, ``$.outputs..[text,label]``), join them with this string into one response. Without it, only the first value is used.
* ``session_json_field`` - (optional) JSONPath to the session or chat id in the endpoint's JSON response. When set, multi-turn conversations stay on one session: the id goes back in the next turn through ``$SESSION`` (see below).
* ``preflight_command`` - (optional) Command run before the first request (a list is run as is, a string through the shell). Nothing is sent unless it exits 0 and, if set, its output matches ``preflight_expect``.
* ``preflight_expect`` - (optional) Regex the preflight output must match.
* ``preflight_every_request`` - (optional) Run the preflight before every request, not only the first. Default ``False``.
* ``preflight_timeout`` - (optional) Seconds before the preflight command counts as failed. Default 60.
* ``stop_patterns`` - (optional) Regexes searched in the raw response body. A match writes ``stop_file`` and stops the run.
* ``stop_file`` - Required with ``stop_patterns``. While this file exists, nothing is sent to the target, in this run or the next ones.

Templates can be either a string or a JSON-serialisable Python object.
Instance of ``$INPUT`` here are replaced with the prompt; instances of ``$KEY``
are replaced with the specified API key. If no key is needed, just don't
put ``$KEY`` in a template.

The ``$INPUT`` and ``$KEY`` placeholders can also be specified in header values.

Multi-turn probes (``goat``, ``fitd``, ``agent_breaker``) need the target to
remember the earlier turns. By default only the last message is sent. For an
endpoint that keeps the chat by an id it returns, set ``session_json_field`` to
that id and put ``$SESSION`` where the request carries it, e.g.
``"req_template_json_object": {"message": "$INPUT", "chat_id": "$SESSION"}``
with ``"session_json_field": "$.chat_id"``. The first turn has no id yet: a
JSON key whose value is exactly ``"$SESSION"`` is left out, and so is a header
holding ``$SESSION``. When a probe goes back to an earlier turn and takes
another way, the earlier user turns are sent again on a new session first.
Run these probes with ``parallel_attempts`` off.

Two guards help when the target can act on real systems (orders, carts,
messages). ``preflight_command`` checks the target is in a safe mode before
anything is sent, e.g. a test or simulation flag in its log; with
``preflight_every_request`` the check repeats before each request, so a
restart into another mode is caught. ``stop_patterns`` catch a response that
shows a real action happened: the run stops with ``TargetStopped`` and
``stop_file`` records the time and the matches. Remove the file only after
looking at what happened.

If we want to call an endpoint where the API key is defined in the value
of an ``X-Authorization`` header, sending and receiving JSON where the prompt
and response value are both under the ``text`` key, we'd define the service
using something like:

.. code-block:: JSON

   {
      "rest": {
         "RestGenerator": {
            "name": "example service",
            "uri": "https://example.ai/llm",
            "method": "post",
            "headers": {
               "X-Authorization": "$KEY"
            },
            "req_template_json_object": {
               "text": "$INPUT"
            },
            "response_json": true,
            "response_json_field": "text"
         }
      }
   }

NB. ``response_json_field`` can also be a JSONPath, for JSON responses where
the target text is not in a top level field. It is treated as a JSONPath
when ``response_json_field`` starts with ``$``.

To use this specification with garak, you can either pass the JSON as a
strong option on the command line via ``--generator_options``, or save the
JSON definition into a file and pass the filename to
``--generator_option_file`` / ``-G``. For example, if we save the above
JSON into ``example_service.json``, we can invoke garak as:

.. code-block::

   garak --target_type rest -G example_service.json

This will load up the default ``RestGenerator`` and use the details in the
JSON file to connect to the LLM endpoint.

If you need something more flexible, add a new module or class and inherit
from RestGenerator.

mTLS Client Certificate Authentication
---------------------------------------

To authenticate with a server that requires mutual TLS, provide your client certificate and (optionally)
key. For a server-side CA bundle, use ``verify_ssl`` with a file path:

.. code-block:: JSON

   {
      "rest": {
         "RestGenerator": {
            "name": "mTLS protected service",
            "uri": "https://api.example.com/llm",
            "client_cert": "/path/to/client.crt",
            "client_key": "/path/to/client.key",
            "client_key_passphrase_env_var": "MTLS_KEY_PASSPHRASE",
            "verify_ssl": "/path/to/ca.crt"
         }
      }
   }

Set the passphrase securely via environment variable:

.. code-block:: bash

   export MTLS_KEY_PASSPHRASE="your_key_passphrase"
   garak --target_type rest -G mtls_config.json --spec probes.dan

If using a combined PEM file containing both certificate and key, omit ``client_key``:

.. code-block:: JSON

   {
      "rest": {
         "RestGenerator": {
            "uri": "https://api.example.com/llm",
            "client_cert": "/path/to/combined.pem",
            "verify_ssl": "/path/to/ca.crt"
         }
      }
   }

----

.. automodule:: garak.generators.rest
   :members:
   :undoc-members:
   :show-inheritance:
