TMP_COMMAND = 'tmpdir=$(mktemp -d -p /tmp "mytmpdir.XXXXXXXXXX" || mktemp -d -t "mytmpdir.XXXXXXXXXX")'
CD_COMMAND = 'cd $tmpdir; echo "tmpdir: $PWD"'


def export_env_command(env_vars: dict) -> str:
    return "export " + " ".join(
        [f"{name}={value}" for name, value in env_vars.items()]
    )
