TMP_COMMAND = 'tmpdir=$(mktemp -d -p /tmp "mytmpdir.XXXXXXXXXX" || mktemp -d -t "mytmpdir.XXXXXXXXXX")'
CD_COMMAND = 'cd $tmpdir; echo "tmpdir: $PWD"'


