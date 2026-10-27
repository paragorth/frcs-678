#!/usr/bin/env bash
#SBATCH --job-name=p1ctl
#SBATCH --account=TORABI-SL4-CPU
#SBATCH --partition=desktop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:30:00
#SBATCH --array=0-28
#SBATCH --output=runs/_slurm/%x-%A_%a.out
# Paper 1's controls from the programme review (27 Oct). Pythia only, cached weights, no downloads.
#   0-9   conflict vs input distribution at 160m (3e-6, 100 steps): maskcopy (same inputs, no conflict), wikianti (conflict
#         on other inputs), anti at doses 0.1 / 0.25 / 0.5 — seeds 0 and 1        [review hole 5]
#   10    copy-with-offset at 160m, wiring logged (3e-6, 400 steps)                [hole 9]
#   11-13 planted damage, one part trained (anti, 1e-4, 300 steps): L4 MLP out, L4 attention qkv, L3 attention out;
#         then attribution with MLP neuron groups and the restore controls        [hole 6, positive control]
#   14-15 SGD with momentum (anti, 3e-3 and 1e-2, 300 steps), then attribution + restore controls   [hole 6]
#   16-20 attribution with 16 MLP neuron groups + restore controls on the existing anti50 models:
#         160m seeds 0-2, 410m, 1.4b                                               [holes 2 and 6]
#   21-23 attribution with tail-robust targets (ce_correct, margin) at 160m seeds 0-2   [hole 7]
#   24-26 410m: maskcopy, wikianti (100 steps), offset (400 steps)                 [holes 5, 9]
#   27-28 SGD stopped at damage matched to Adam's anti50 (3e-3 for 30 steps, 1e-2 for 10: copying ~0.02, perplexity
#         +65%), then attribution + restore controls; the 300-step arms 14-15 wrecked the model (perplexity
#         9 000-42 000), so their restore test is not a fair match                  [hole 6, added 27 Oct]
set -euo pipefail
if [ -z "${SEMATAXY_IN_SRUN:-}" ]; then
  export SEMATAXY_IN_SRUN=1
  exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$0" "$@"
fi
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm runs/pythia_damaged results/live_learning
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=$HOME/rds/hpc-work/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
R=results/live_learning; D=runs/pythia_damaged; M160=EleutherAI/pythia-160m; M410=EleutherAI/pythia-410m
ind() { python tools/pythia_induction.py "$@"; }
attr_rc() {   # model damaged tag : attribution with 16 MLP groups, then the restore controls
  [ -s $R/${3}_attribution_g16.csv ] || python tools/pythia_attribution.py --model "$1" --damaged "$2" --mlp-groups 16 --out $R/${3}_attribution_g16.csv
  python tools/restore_controls.py --model "$1" --damaged "$2" --attribution $R/${3}_attribution_g16.csv --out $R/${3}_restore_controls.csv
}
CM="--lr 3e-6 --steps 100 --marks 0,5,10,20,30,50,75,100"
i=$SLURM_ARRAY_TASK_ID; echo "p1ctl $i on $(hostname) at $(date +%H:%M)"
case $i in
  0) ind --model $M160 --task maskcopy $CM --seed 0 --out $R/conflictctl_160m_maskcopy_s0.csv ;;
  1) ind --model $M160 --task maskcopy $CM --seed 1 --out $R/conflictctl_160m_maskcopy_s1.csv ;;
  2) ind --model $M160 --task wikianti $CM --seed 0 --out $R/conflictctl_160m_wikianti_s0.csv ;;
  3) ind --model $M160 --task wikianti $CM --seed 1 --out $R/conflictctl_160m_wikianti_s1.csv ;;
  4) ind --model $M160 --task anti --anti-frac 0.1 $CM --seed 0 --out $R/conflictctl_160m_anti0.1_s0.csv ;;
  5) ind --model $M160 --task anti --anti-frac 0.25 $CM --seed 0 --out $R/conflictctl_160m_anti0.25_s0.csv ;;
  6) ind --model $M160 --task anti --anti-frac 0.5 $CM --seed 0 --out $R/conflictctl_160m_anti0.5_s0.csv ;;
  7) ind --model $M160 --task anti --anti-frac 0.1 $CM --seed 1 --out $R/conflictctl_160m_anti0.1_s1.csv ;;
  8) ind --model $M160 --task anti --anti-frac 0.25 $CM --seed 1 --out $R/conflictctl_160m_anti0.25_s1.csv ;;
  9) ind --model $M160 --task anti --anti-frac 0.5 $CM --seed 1 --out $R/conflictctl_160m_anti0.5_s1.csv ;;
  10) ind --model $M160 --task offset --lr 3e-6 --steps 400 --marks 0,10,20,50,100,200,300,400 --seed 0 --out $R/offsetwiring_160m_s0.csv ;;
  11|12|13)
    P=(x x x x x x x x x x x layers.4.mlp.dense_4h_to_h layers.4.attention.query_key_value layers.3.attention.dense)
    N=(x x x x x x x x x x x L4mlpout L4qkv L3attnout)
    p=${P[$i]}; n=${N[$i]}
    [ -s $D/plant_${n}.pt ] || ind --model $M160 --task anti --train-only "$p" --lr 1e-4 --steps 300 --marks 0,10,30,100,200,300 \
        --seed 0 --save-damaged $D/plant_${n}.pt --out $R/plant_160m_${n}.csv
    attr_rc $M160 $D/plant_${n}.pt plant_${n} ;;
  14|15)
    LR=(x x x x x x x x x x x x x x 3e-3 1e-2); lr=${LR[$i]}
    [ -s $D/sgd_${lr}.pt ] || ind --model $M160 --task anti --optim sgd --lr $lr --steps 300 --marks 0,10,30,50,100,200,300 \
        --seed 0 --save-damaged $D/sgd_${lr}.pt --out $R/sgd_160m_${lr}.csv
    attr_rc $M160 $D/sgd_${lr}.pt sgd_${lr} ;;
  16) attr_rc $M160 $D/anti50.pt pythia160m ;;
  17) attr_rc $M160 $D/anti50_160m_s1.pt pythia160m_s1 ;;
  18) attr_rc $M160 $D/anti50_160m_s2.pt pythia160m_s2 ;;
  19) attr_rc $M410 $D/anti50_410m.pt pythia410m ;;
  20) attr_rc EleutherAI/pythia-1.4b $D/anti50_1.4b.pt pythia1.4b ;;
  21|22|23)
    DM=(x x x x x x x x x x x x x x x x x x x x x anti50.pt anti50_160m_s1.pt anti50_160m_s2.pt)
    TG=(x x x x x x x x x x x x x x x x x x x x x pythia160m pythia160m_s1 pythia160m_s2)
    for t in ce_correct margin; do
      python tools/pythia_attribution.py --model $M160 --damaged $D/${DM[$i]} --target $t --out $R/${TG[$i]}_attribution_${t}.csv
    done ;;
  24) ind --model $M410 --task maskcopy $CM --seed 0 --out $R/conflictctl_410m_maskcopy_s0.csv ;;
  25) ind --model $M410 --task wikianti $CM --seed 0 --out $R/conflictctl_410m_wikianti_s0.csv ;;
  26) ind --model $M410 --task offset --lr 3e-6 --steps 400 --marks 0,10,20,50,100,200,300,400 --seed 0 --out $R/offsetwiring_410m_s0.csv ;;
  27) [ -s $D/sgd_3e-3_s30.pt ] || ind --model $M160 --task anti --optim sgd --lr 3e-3 --steps 30 --marks 0,10,20,30 \
        --seed 0 --save-damaged $D/sgd_3e-3_s30.pt --out $R/sgd_160m_3e-3_s30.csv
      attr_rc $M160 $D/sgd_3e-3_s30.pt sgd_3e-3_s30 ;;
  28) [ -s $D/sgd_1e-2_s10.pt ] || ind --model $M160 --task anti --optim sgd --lr 1e-2 --steps 10 --marks 0,5,10 \
        --seed 0 --save-damaged $D/sgd_1e-2_s10.pt --out $R/sgd_160m_1e-2_s10.csv
      attr_rc $M160 $D/sgd_1e-2_s10.pt sgd_1e-2_s10 ;;
esac
echo "p1ctl $i done at $(date +%H:%M)"
