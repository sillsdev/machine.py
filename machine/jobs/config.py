from pathlib import Path
from typing import cast

from dynaconf import Dynaconf
from dynaconf.base import Settings

CONFIG_DIR = Path(__file__).parent

SETTINGS = cast(
    Settings, Dynaconf(envvar_prefix="MACHINE", settings_files=[str(CONFIG_DIR / "settings.yaml")], environments=True)
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
