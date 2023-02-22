#! /usr/bin/env bash

launch_devcontainer() {

    # Ensure we're in the repo root
    cd "$( cd "$( dirname "$(realpath -m "${BASH_SOURCE[0]}")" )" && pwd )/..";

    local pkgs="${1:-conda}";
    local mode="${2:-single}";

    case "$pkgs" in
        pip   ) ;;
        conda ) ;;
        *     ) pkgs="conda";;
    esac

    case "$mode" in
        single  ) ;;
        unified ) ;;
        isolated) ;;
        *      ) mode="single";;
    esac

    local tmpdir="$(mktemp -d)";
    local flavor="devcontainer-${pkgs}/${mode}";
    local path="$(pwd)/.devcontainer/${flavor}";

    if [[ "$mode" == "isolated" ]]; then
        cp -arL "$path/.devcontainer" "${tmpdir}/";
        sed -i "s@\${localWorkspaceFolder}@$(pwd)@g" "${tmpdir}/.devcontainer/devcontainer.json";
        path="${tmpdir}";
    fi

    local hash="$(echo -n "${path}" | xxd -pu - | tr -d '[:space:]')";
    local url="vscode://vscode-remote/dev-container+${hash}/home/coder";

    echo "devcontainer URL: $url";

    local launch="";
    if type open >/dev/null 2>&1; then
        launch="open";
    elif type xdg-open >/dev/null 2>&1; then
        launch="xdg-open";
    fi

    if [ -n "${launch}" ]; then
        code --new-window "$tmpdir";
        $launch "$url" >/dev/null 2>&1 &
    fi
}

launch_devcontainer "$@";
