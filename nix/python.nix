{ pkgs ? import <nixpkgs> { } }:

let
  pypkgs = pkgs.python3.withPackages (
    pythonPackages: with pythonPackages; [
      numpy
      scipy
      notebook
      matplotlib
      pandas
      # Open Source Tool to Download Yahoo! Financial Data
      yfinance
      # ML part
      scikit-learn
    ]
  );
in
pkgs.mkShell {
  packages = [
    pypkgs
  ];

  shellHook = ''
    echo "Python development environment"
    echo "Python: $(python --version)"
    echo "Start Jupyter with: jupyter notebook"
  '';
}
