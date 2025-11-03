{ pkgs ? import <nixpkgs> {} }:

let
  pythonEnv = pkgs.python311.withPackages (ps: with ps; [
    pytelegrambotapi
    requests
    schedule
    pytz
    sentencepiece
    tiktoken
    openai-whisper
    numpy
    transformers
    sentence-transformers
    torch
    torchaudio
    scikit-learn
    scipy
    faiss
  ]);
in

pkgs.mkShell {
  name = "telegram-ai-bot-shell";

  buildInputs = [
    pythonEnv
    pkgs.ffmpeg
  ];
}
