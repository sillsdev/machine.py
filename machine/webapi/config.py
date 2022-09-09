from typing import cast

from dynaconf import Dynaconf
from dynaconf.base import Settings

settings = cast(
    Settings, Dynaconf(envvar_prefix="DYNACONF", settings_files=["machine/webapi/settings.yaml"], environments=True)
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
