#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-katarank100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/katarank/pilot-%j.out
set -euo pipefail
ulimit -c 0
assets="$PWD/data/external/katarank_nano4"
out="$PWD/outputs/rank/katarank/$SLURM_JOB_ID"
mkdir "$out"
if [[ -n "${KATARANK_BUILD_FROM:-}" ]]; then
    previous="$PWD/outputs/rank/katarank/$KATARANK_BUILD_FROM"
    test -x "$previous/build/katago"
    sha256sum --check "$previous/source.sha256" > "$out/reused-build.log"
    ln -s "$previous/rank" "$out/rank"
    ln -s "$previous/build" "$out/build"
    cp "$previous/source.sha256" "$out/source.sha256"
else
    for name in rank engine; do
        mkdir "$out/$name"
        tar -xzf "$assets/$name.tar.gz" -C "$out/$name" --strip-components=1
    done
    sha256sum "$assets"/{rank.tar.gz,engine.tar.gz,*.bin.gz} > "$out/source.sha256"
    cmake -S "$out/engine/cpp" -B "$out/build" \
        -DCMAKE_BUILD_TYPE=Release -DUSE_BACKEND=OPENCL -DBUILD_DISTRIBUTED=OFF \
        -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so \
        > "$out/build.log" 2>&1
    cmake --build "$out/build" --parallel "$SLURM_CPUS_PER_TASK" >> "$out/build.log" 2>&1
fi
if [[ -n "${KATARANK_TUNING_FROM:-}" ]]; then
    tuning="$PWD/outputs/rank/katarank/$KATARANK_TUNING_FROM/engine-home/opencltuning"
    mkdir "$out/engine-home"
    cp -r "$tuning" "$out/engine-home/"
fi
nvidia-smi --query-gpu=timestamp,name,driver_version,memory.used,utilization.gpu --format=csv -l 1 > "$out/gpu.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT
export PYTHONPATH="$PWD/src:$PWD:$out/rank/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/jason1122/.conda/envs/go/bin/python \
    -m scripts.strength.rank_katarank_pilot --out "$out" \
    --rank-source "$out/rank" --engine "$out/build/katago" \
    --network "$assets/kata1-b28c512nbt-s12763923712-d5805955894.bin.gz" --threads "$SLURM_CPUS_PER_TASK"
