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

          workspace =
            uv2nix.lib.workspace.loadWorkspace {
              workspaceRoot = ./.;
            };

          pythonBase =
            pkgs.callPackage
              pyproject-nix.build.packages
              {
                inherit python;
              };

          overlay =
            workspace.mkPyprojectOverlay {
              sourcePreference = "wheel";
            };
          
          sktimeFixOverlay = final: prev: {
            sktime = prev.sktime.overrideAttrs (old: {
              postInstall = (old.postInstall or "") + ''
                rm -rf $out/lib/python${python.pythonVersion}/site-packages/docs
              '';
            });
          };
          
          pythonSet =
            pythonBase.overrideScope (
              nixpkgs.lib.composeManyExtensions [
                pyproject-build-systems.overlays.wheel
                overlay
                sktimeFixOverlay
              ]
            );
          
          virtualenv =
            pythonSet.mkVirtualEnv
              "grifops-env"
              workspace.deps.default;

        in
        {
          default = pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
            ];

            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };
            
            shellHook = ''
              unset PYTHONPATH
            
              echo "GRIFOps development environment"
              echo "Python: $(python --version)"
            '';
          };
        }
      );
    };
}
