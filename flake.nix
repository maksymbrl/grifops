{
  description = "GRIFOps Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems =
        nixpkgs.lib.genAttrs supportedSystems;

    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };

          python = pkgs.python3;

          # Load pyproject.toml and uv.lock as a uv workspace.
          workspace =
            uv2nix.lib.workspace.loadWorkspace {
              workspaceRoot = ./.;
            };

          # Base Python package set used by pyproject.nix.
          pythonBase =
            pkgs.callPackage
              pyproject-nix.build.packages
              {
                inherit python;
              };

          # Convert uv.lock dependencies into a Nix overlay.
          # Prefer binary wheels where available.
          overlay =
            workspace.mkPyprojectOverlay {
              sourcePreference = "wheel";
            };

          # Temporary fix for the sktime wheel, which contains a
          # top-level docs directory that conflicts during installation.
          sktimeFixOverlay =
            final: prev: {
              sktime = prev.sktime.overrideAttrs (old: {
                postInstall = (old.postInstall or "") + ''
                  rm -rf \
                    $out/lib/python${python.pythonVersion}/site-packages/docs
                '';
              });
            };

          # Build the regular Python package set from uv.lock.
          pythonSet =
            pythonBase.overrideScope (
              nixpkgs.lib.composeManyExtensions [
                pyproject-build-systems.overlays.wheel
                overlay
                sktimeFixOverlay
              ]
            );

          # Make workspace packages editable during development.
          #
          # Instead of copying GRIFOps source files into /nix/store,
          # the environment points back to the local working tree.
          editableOverlay =
            workspace.mkEditablePyprojectOverlay {
              root = "$REPO_ROOT";
            };

          editablePythonSet =
            pythonSet.overrideScope editableOverlay;

          # Development virtual environment.
          #
          # workspace.deps.all also includes dependency groups such as
          # future development dependencies (e.g. pytest).
          virtualenv =
            editablePythonSet.mkVirtualEnv
              "grifops-env"
              workspace.deps.all;

        in
        {
          default =
            pkgs.mkShell {
              packages = [
                virtualenv
                pkgs.uv
                pkgs.git
              ];

              env = {
                # uv2nix owns the Python environment.
                # Do not let uv create/synchronize a separate .venv.
                UV_NO_SYNC = "1";

                # Tell uv which Nix-provided Python interpreter to use.
                UV_PYTHON =
                  editablePythonSet.python.interpreter;

                # Never download a Python interpreter through uv.
                UV_PYTHON_DOWNLOADS = "never";
              };

              shellHook = ''
                # Avoid PYTHONPATH inherited from Nix Python builders.
                unset PYTHONPATH

                # Used by mkEditablePyprojectOverlay above.
                export REPO_ROOT=$(git rev-parse --show-toplevel)

                echo "GRIFOps development environment"
                echo "Repository: $REPO_ROOT"
                echo "Python: $(python --version)"
                echo "Python executable: $(which python)"
              '';
            };
        }
      );
    };
}
