# Large fits on CPU clusters

How to size `gmri decompose` / `gmri replicability` jobs when the input is large: voxel input, or anything from ~10⁵ columns per subject. ROI input (a few hundred labels) needs none of this, and the defaults are right for it.

The numbers below were measured on a laptop (Intel Core Ultra 7 155H) with 30 subjects × 4 time points and ranks 2–4, then extrapolated. Treat them as starting points. Measure on your own data and cluster first: every run records what each restart cost (`restarts.csv`), and the [last section](#running-the-scaling-test-on-a-cluster) describes the full test.

All settings below are environment variables (see [cli.md](cli.md#cpu-and-gpu)). They are read when the fits run, so each job can use its own without a new `plan`.

## How the cost grows

- **Linearly with the data.** A fit's time per iteration is proportional to subjects × time points × voxels, from ~10⁵ voxels up. At 30 × 4 × 11 M voxels, rank 3, one iteration takes:

  | method | numpy, 1 thread | torch, 4 threads |
  |---|---|---|
  | PARAFAC2 / CMF | ~17–29 s | ~6–10 s |
  | CP | ~23–33 s | ~9–13 s |

  The lower end of each range is a restart alone on its node; the upper end is one on a busy node.
- **With rank**, roughly in proportion: rank 4 costs ~1.3–2× rank 2.
- **With the iteration count.** A restart stops between a few hundred iterations and `--max-iter` (2000), depending on the data and the rank. Plan for the cap unless you have measured otherwise.
- **Memory per restart**, at 11 M voxels:
  - **On its own:** about 20 GB for PARAFAC2/CMF and 33 GB for CP. In general, ~1 GB plus 3.5–4× (PARAFAC2/CMF) or 6–7× (CP) the size of the input in float32. The input in float32 is subjects × time points × voxels × 4 bytes, so 5.3 GB here.
  - **As a worker of a pool** (several restarts per job, below): the workers share one float64 copy of the input (10.6 GB) and add only their own temporaries, ~4–7 GB each for PARAFAC2/CMF and ~10–17 GB for CP.

## Threads: `CPUS_PER_TASK` and the fit backend

`CPUS_PER_TASK` sets the threads of each job, shared out over its `--restart-procs` workers. Whether they help depends on the backend, `GMRITENSOR_FIT_BACKEND`:

| backend | best for | threads worth giving one fit |
|---|---|---|
| `numpy` (default) | ROI input, small tensors (≲ 10⁵ voxels): 2–5× faster than torch there | **1**. Extra threads gain ≤ 1.3×, because most of the work is single-threaded numpy; a job warns when its numpy fits get more than one. |
| `torch` | voxel input (≳ 5·10⁵ voxels): 1.1–1.9× faster than numpy at 1 thread | **about 4**. That gives ~2–2.5×; 8 or more threads gain little. |
| `auto` | mixed inputs | numpy below `GMRITENSOR_AUTO_BACKEND_ENTRIES` data entries (default 5·10⁷, about 30 × 4 × 4·10⁵ voxels), torch above it. |

Every fit runs in float64 on both backends, so the backend changes the speed and not the result (beyond round-off).

Use physical cores (`#SBATCH --hint=nomultithread`): hyper-threads do not speed these fits up.

## How many restarts per node

Restarts are independent, so a node should run as many of them side by side as fit. Two things set the limit:

1. **RAM**: node RAM ÷ memory per restart. For voxel input in one-restart jobs this is the binding limit: a 512 GB node fits ~24 PARAFAC2/CMF restarts at 11 M voxels, or ~14 CP restarts. As pool workers they need far less, so a node fits 2–3× more.
2. **Memory bandwidth.** These fits move a lot of data per operation, so a node is "full" before all its cores are busy. Each single-threaded fit draws up to one core's streaming bandwidth (~4–12 GB/s), so a 300 GB/s node saturates at about 25–75 single-threaded restarts. Past that point, adding restarts only slows them all down.

Give the cores left over to threads, on the torch backend: for example, 24 restarts × 4 threads on a 128-core node.

## One restart per job, or several

`plan --tasks-per-job N` puts N restarts in each job. `--restart-procs P` (`--n-procs` for `replicability`) runs P of them at a time in one process pool. Each pool worker gets `CPUS_PER_TASK ÷ P` threads.

| | one restart per job (`--tasks-per-job 1`) | several per job (`--tasks-per-job N --restart-procs P`) |
|---|---|---|
| CPU time | same | same |
| start-up and loading | every restart imports the package (~7 s) and loads `voxels.parquet` | once per job |
| memory | each restart holds the input in float32 and float64 (16 GB at 11 M voxels) | the job holds them once (16 GB); each worker adds only its temporaries |
| needs | — | `/dev/shm` at least the size of the float64 input (10.6 GB at 11 M voxels); it usually counts toward the job's memory |
| scheduling | small jobs fill gaps in the queue | needs whole nodes, or large slices of them |

Both give the same results. Several restarts per job make far better use of a node's memory; one restart per job is easier to fit into a busy queue.

Containers: Docker's default `/dev/shm` is 64 MB, so a pool fails with "No space left on device" for anything bigger. Apptainer uses the host's `/dev/shm`.

## SLURM template (voxel input, several restarts per job)

```bash
# 4 restarts per job, 4 threads each: 16 cores; 16 GB for the input + 4 x ~7 GB, plus headroom
N=$(gmri decompose plan --input results/data/voxels.parquet --output-dir results/parafac2 \
        --method parafac2 --ranks 2 3 4 --restarts 50 --tasks-per-job 4 --restart-procs 4)
sbatch --array=0-$((N-1)) job.sh
```

`job.sh`:

```bash
#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
export CPUS_PER_TASK=$SLURM_CPUS_PER_TASK   # split over --restart-procs: 4 threads each
export GMRITENSOR_FIT_BACKEND=torch          # voxel input; keep numpy for ROI input
gmri decompose run --output-dir results/parafac2 --job $SLURM_ARRAY_TASK_ID
```

For one restart per job: `--tasks-per-job 1 --restart-procs 1`, `--cpus-per-task=4` and `--mem` ≈ the memory of one restart plus a few GB.

`--time` must cover `--tasks-per-job ÷ --restart-procs` restarts one after another. With the numbers above, plan for 4–7 h per PARAFAC2/CMF restart at the iteration cap (torch, 4 threads). An interrupted job can be resubmitted: it skips the restarts it already stored (see [cli.md](cli.md#distributed-restarts-slurm)).

## Reading `restarts.csv`

Every `decompose run` and `collect` writes `<store>/<method>_rank_<r>/restarts.csv`, one row per restart:

| column | meaning |
|---|---|
| `status` | `ok`, or why the restart was rejected (`max_iter`, `reconstruction`, `feasibility`, `degenerate`) |
| `n_iter` | iterations the fit ran |
| `fit_seconds`, `seconds_per_iter` | the fit alone: no start-up or data loading |
| `backend`, `threads` | where it ran: `numpy`, `torch` or `cuda`, and the threads it had |
| `peak_rss_gb` | peak memory of the process that fitted it, up to the end of this fit (in a pool, also its earlier restarts) |
| `host` | the node |

`seconds_per_iter` × the iterations a converged restart needs gives the time per restart. Rejected restarts are timed too, so a short run with `--max-iter 20` measures the speed of a configuration even though its fits are not kept.

## Running the scaling test on a cluster

The laptop numbers above say which way the choices go. The best settings for a given cluster and dataset come from a short test on it. The steps below use only `gmri` and `restarts.csv`.

1. **Choose the test data.** Use the real input, or a voxel subset of it with the same subjects and time points. Cost per iteration is linear in voxels from ~10⁵ on, so a 10 % subset is enough to compare configurations, and the full input checks memory.
2. **Measure speed per configuration.** One job per configuration, each with its own `--output-dir`, on otherwise idle nodes of the target partition:

   | variable | values to try |
   |---|---|
   | `GMRITENSOR_FIT_BACKEND` | `numpy`, `torch` |
   | threads per restart (`CPUS_PER_TASK ÷ --restart-procs`) | 1, 2, 4, 8 |
   | `--restart-procs` | 1, and as many as one node's RAM allows |
   | `--method`, `--ranks` | the ones you will fit |

   ```bash
   export GMRITENSOR_FIT_BACKEND=torch CPUS_PER_TASK=16
   gmri decompose run --input <voxels.parquet> --output-dir scaling/torch_t4_p4 \
       --method parafac2 --ranks 3 --restarts 4 --restart-procs 4 --max-iter 20
   ```

   Such short runs may reject every restart and end with "no decomposition converged" (exit code 2). That is expected: `restarts.csv` is still written, in `scaling/torch_t4_p4/restarts/parafac2_rank_3/`.
3. **Measure iteration counts.** Run one configuration with the real `--max-iter` (2000) and a few restarts per rank, and read `n_iter` of the `ok` restarts.
4. **Read the results.** For each configuration, compute from `restarts.csv`:
   - **speed:** the median `seconds_per_iter`;
   - **memory:** the maximum `peak_rss_gb`;
   - **throughput:** (restarts running at once) ÷ (`seconds_per_iter` × iterations per restart), in restarts per node-hour.

   Then:
   - Compare `--restart-procs 1` with a full node: the slowdown in `seconds_per_iter` shows the memory-bandwidth limit.
   - Repeat at two input sizes to place the numpy/torch crossover, and set `GMRITENSOR_AUTO_BACKEND_ENTRIES` to it.
5. **Report back:**
   - the crossover size;
   - the best threads per restart for each backend;
   - restarts per node and memory per restart, at the real size;
   - `seconds_per_iter` and iterations per restart, for each method and rank.

   From these, set `--tasks-per-job`, `--restart-procs`, `CPUS_PER_TASK`, `--mem` and `--time` for the production array.
