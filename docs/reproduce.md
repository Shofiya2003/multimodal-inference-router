# Setup, Reproduction and Repository Layout

How the experiments were set up, how to rerun them on AWS, and where every script and result file lives.

## 1. Experimental Setup

**Fixed for all runs.** Every node serves Qwen2.5-Omni-3B on CPU in llama.cpp's `llama-server` (commit `8642541b2`), with Q8_0 weights unless a run varies the quantization, and the image/audio encoder always in Q8_0. Each server slot has 4,096 tokens of context. Every request uses temperature 0, at most 128 output tokens, and no prompt cache, so the same input gives the same answer on every node. All instances run in AWS us-west-2 on Ubuntu 24.04 or later, and the benchmark client runs on the same instance as the server, so network delay is not part of the measurements.

**Workloads.** Four request types in nine sizes, each from a public dataset: chat with a short question (Dolly-15k), text summarization of 512, 1,024, or 3,072 tokens (CNN/DailyMail), image description of 1, 2, or 4 images (COCO 2017 validation), and transcription of 10 or 40 seconds of audio (Mini LibriSpeech). Chat time is mostly spent generating the answer (decode), while long text, images, and audio are mostly spent processing the input (prefill, plus encoding for images and audio). The full workload table and prompts are in [`docs/baseline-benchmark.md`](baseline-benchmark.md) (section 1).

**Baseline benchmarks.** The baseline is a c7i.2xlarge (Intel Sapphire Rapids with AMX, 8 vCPUs, 16 GB) serving Q8_0. Every other configuration changes exactly one thing from it:

| Variable     | Configurations compared with the baseline                                   |
| ------------ | ---------------------------------------------------------------------------- |
| Architecture | c7a.2xlarge (AMD Genoa), c8g.2xlarge (ARM Graviton4), c7i.2xlarge with AMX off, c6i.2xlarge (Intel Ice Lake, no AMX) |
| vCPUs        | c7i.xlarge (4 vCPUs, 8 GB), c7i.4xlarge (16 vCPUs, 32 GB)                    |
| RAM          | m7i.2xlarge (32 GB), r7i.2xlarge (64 GB)                                     |
| Quantization | Q4_K_M (4-bit), f16 (16-bit), both on c7i.2xlarge                            |

The vCPU rows change RAM along with vCPUs, because EC2 ties them within a family. A subset of these nodes also runs the concurrency and queueing sweep. The full matrix, with physical cores, prices, and the results folder for each configuration, is in [`docs/baseline-benchmark.md`](baseline-benchmark.md) (section 2).

**Routing benchmarks.** One node per CPU family from the baseline benchmarks, all serving Q8_0 with the same llama.cpp build:

| Node        | CPU                               | vCPUs / RAM | Slots |
| ----------- | --------------------------------- | ----------- | ----- |
| c7a.2xlarge | AMD Genoa                         | 8 / 16 GB   | 2     |
| c7i.xlarge  | Intel Sapphire Rapids, AMX        | 4 / 8 GB    | 1     |
| m8g.xlarge  | ARM Graviton4                     | 4 / 16 GB   | 2     |

The m8g.xlarge has the same Graviton4 CPU as the c8g but 16 GB of memory, and the c7i.xlarge uses one slot because two large image requests at once ran it out of memory. Reasons for each choice, the traffic traces, and prices are in [`docs/routing-benchmark.md`](routing-benchmark.md) (sections 2 and 3).

## 2. Running the Benchmarks

### Requirements

- An AWS account in us-west-2 with at least 16 vCPUs of on-demand quota (the routing cluster uses all 16; stopped instances do not count).
- Ubuntu Server 24.04 LTS (64-bit x86 for Intel and AMD, ARM for Graviton), a 30 GiB gp3 disk, an SSH key pair, and inbound SSH from your IP only.
- On each instance: Python 3 with a virtual environment (`requirements.txt`) and about 3 GB of datasets. `run.sh` installs the build tools and builds llama.cpp itself.
- Time and cost: a full baseline sweep takes about 5–6 hours per node; the routing benchmarks take about 40 minutes of setup, 1–1.5 hours of profiling, and about 1.7 hours of runs, at about $0.77/hour for the cluster.

### Baseline benchmarks (one node at a time)

On the instance, inside tmux so the run continues if SSH disconnects:

```bash
git clone https://github.com/Shofiya2003/multimodal-inference-router && cd multimodal-inference-router
./src/run.sh qwen                                   # builds llama.cpp, downloads Q8_0 + encoder, starts llama-server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/loaders/dolly.py --download && python src/loaders/cnn_dailymail.py --download && python src/loaders/coco.py --download && python src/loaders/librispeech.py --download
tmux new -s bench
NODE=<label> ./src/experiment.sh qwen               # full sweep, ~5–6 h; results in results/phase1/<label>__<model>/
```

Copy the results folder off the instance (`scp -r`) before terminating it; stopping it keeps the disk. Variants (quantization rows, AMX off, the 8 GB c7i.xlarge, answer quality) are in [`docs/baseline-benchmark.md`](baseline-benchmark.md) (section 5).

### Routing benchmarks (3-node cluster)

These steps match the runs in `results/routing/`; [`docs/routing-cluster-runbook.md`](routing-cluster-runbook.md) has the full commands for the cluster setup:

1. **Network and security groups.** All three nodes must be in the same VPC and availability zone. Allow SSH from your IP to each node, and SSH from the c7a's security group to the other two, so the router can reach them; port 8080 stays closed.

2. **Start the nodes.** Start or launch the c7a.2xlarge, c7i.xlarge, and m8g.xlarge (ARM image, 30 GiB) in the c7a's subnet, with shutdown behavior set to **Stop**. Note each node's public and private IP; public IPs change on every start.

3. **Set up each node.** Clone the repo, start the server with `./src/run.sh qwen` (2 slots), create the venv, and download the datasets, as in the baseline benchmarks.

4. **Profile each node** in tmux, with nothing else running on it (~1–1.5 h, all three in parallel):
   ```bash
   L=<node>-p2      # c7a.2xlarge-p2, c7i.xlarge-p2, or m8g.xlarge-p2
   python src/benchmark.py --node $L --repeats 2 --concurrency 1 --output natural && \
   python src/benchmark.py --node $L --repeats 2 --concurrency 1 2 --output fixed
   ```
   The first command gives `ect` its time per workload; the second gives each node's slot gain.

5. **Configure the c7i.xlarge for its 8 GB of memory.** Restart its server with one slot and the prompt cache off (`llama-server ... -np 1 --cache-ram 0`; the full command is in step 5 of the runbook). With two slots, two large image requests at once ran it out of memory, and llama-server's default 8 GB prompt cache did too under 90% load.

6. **Set up the router on the c7a.** Create a router SSH key and add it to the other nodes' `authorized_keys`, add `node-c7a`, `node-c7i`, and `node-m8g` entries to `~/.ssh/config` (private IPs), open SSH tunnels to the other two servers in tmux (ports 8081 and 8082), and check all three `/health` endpoints. Then write `nodes.json` as in [`results/routing/nodes.json`](../results/routing/nodes.json): the entry order is the round-robin order, and `slot_gain` is set to 1.0 for the c7i.xlarge (one slot) and 1.01 for the m8g.xlarge.

7. **Pick the loads.** Copy the other two profiles to the c7a, then:
   ```bash
   python src/main.py capacity --nodes nodes.json
   python src/tracegen.py phase1 --rate <R60> --duration 900 --seed 1 --out traces/cluster-load60.csv
   python src/tracegen.py phase1 --rate <R90> --duration 900 --seed 1 --out traces/cluster-load90.csv
   ```

8. **Quick test.** `python src/main.py run --trace traces/cluster-load60.csv --nodes nodes.json --policy rr --cluster test --requests 6` should return 6 `ok` requests, two per node; then delete the test output (`rm -rf results/routing`).

9. **Run the three policies at both loads** on the c7a, in tmux (~1.7 h, about 17 minutes per run), then stop all three nodes:
   ```bash
   for load in 60 90; do
     for p in rr least ect; do
       python src/main.py run --trace traces/cluster-load$load.csv --nodes nodes.json --policy $p --cluster c7a-c7i-m8g \
         2>&1 | tee -a results/routing-console.log
     done
   done; \
   ssh node-c7i sudo shutdown -h now; ssh node-m8g sudo shutdown -h now; sudo shutdown -h now
   ```

10. **Collect the results.** Start only the c7a and copy `results/routing/`, the `results/phase1/*-p2__*` profiles, `traces/`, and the console log back with `scp`. Then stop it again, and terminate the nodes once nothing more needs to run.

Policies, metrics, and result files: [`docs/routing-benchmark.md`](routing-benchmark.md) (sections 4–6).

## 3. Repository Layout

**Server and baseline benchmarks**

- [`run.sh`](../src/run.sh): builds llama.cpp (commit `8642541b2`, CPU only), downloads Qwen2.5-Omni-3B (`Q8_0` by default, or `Q4_K_M` / `f16`) with its Q8_0 image/audio encoder, and starts `llama-server` in tmux. `SLOTS=4` changes the number of requests it processes at once (default 2).
- [`benchmark.py`](../src/benchmark.py): the single-node benchmark. Runs on the instance next to the server, sends workloads 00–08, samples `llama-server`'s CPU and memory, and writes one row per request and one summary row per workload.
- [`experiment.sh`](../src/experiment.sh): the full concurrency sweep in one command (2 slots at concurrency 1 and 2, the queue at 5 and 10, 4 slots at concurrency 4), with a log and a final request count check.
- [`requirements.txt`](../requirements.txt): Python packages for the client scripts.

**Datasets** (each script downloads its data into `dataset/`, which git ignores)

- [`dolly.py`](../src/loaders/dolly.py): Dolly-15k short questions (chat, run 00).
- [`cnn_dailymail.py`](../src/loaders/cnn_dailymail.py): CNN/DailyMail news articles cut to 512, 1,024, or 3,072 tokens (text, runs 01–03).
- [`coco.py`](../src/loaders/coco.py): COCO 2017 validation images (image, runs 04–06).
- [`librispeech.py`](../src/loaders/librispeech.py): Mini LibriSpeech recordings as 10 s and 40 s clips (audio, runs 07–08).

**Answer quality**

- [`quality.py`](../src/quality.py): collects answers from the served quantization and scores them against f16 (ROUGE-L) and the human transcripts (word error rate).
- [`perplexity.sh`](../src/perplexity.sh): perplexity of each quantization on wikitext-2.

**Routing**

- [`router.py`](../src/router.py): the routing policies (`rr`, `least`, `ect`, and the draft `slo`) and their cost model, built from the node profiles.
- [`main.py`](../src/main.py): estimates cluster capacity (`capacity`), replays a trace over the cluster with one policy (`run`), and simulates every policy on a trace (`sim`).
- [`nodes.example.json`](../nodes.example.json): template for the cluster's node list (`nodes.json`).
- [`traces/`](../traces/): the two traffic traces replayed by every policy, [`cluster-load60.csv`](../traces/cluster-load60.csv) and [`cluster-load90.csv`](../traces/cluster-load90.csv).

**Analysis and docs**

- [`analysis/plot.ipynb`](../analysis/plot.ipynb): plots from `results/phase1/` for architecture, memory, vCPU, quantization, and concurrency, plus a summary figure. The figures are saved in the notebook's outputs.
- [`analysis/quantization/`](../analysis/quantization/): f16 vs. Q8_0 vs. Q4_K_M on the c7i.2xlarge: speed, memory, and answer quality ([`README.md`](../analysis/quantization/README.md)), its figures, and [`make_plots.py`](../analysis/quantization/make_plots.py), which regenerates them.
- [`analysis/routing/`](../analysis/routing/): the routing results write-up ([`README.md`](../analysis/routing/README.md)), its figures, and [`analyze_routing.py`](../analysis/routing/analyze_routing.py), which regenerates them from `results/routing/`.
- [`docs/baseline-benchmark.md`](baseline-benchmark.md): baseline benchmark setup, methodology, metrics, and reproduction steps.
- [`docs/routing-benchmark.md`](routing-benchmark.md): routing benchmark motivation, cluster, traffic, policies, and metrics.
- [`docs/routing-cluster-runbook.md`](routing-cluster-runbook.md): step-by-step setup and runs for the routing cluster.

**Results**

- [`results/phase1/`](../results/phase1/): one folder per node label and quantization, named `<node>__<model>`. `summary.csv` / `requests.csv` (v1) hold one-request-at-a-time runs from the first version of `benchmark.py`; `summary_v2.csv` / `requests_v2.csv` (v2) hold runs from the current version (concurrency sweeps and routing profiles); `logs/` holds `experiment.sh` and `llama-server` output.

  | Folder (`…__Qwen2.5-Omni-3B-<quant>`) | Configuration | Files |
  | ------------------------------------- | ------------- | ----- |
  | `c7i.2xlarge__…-Q8_0` | Baseline; Q8_0 concurrency sweep | v1, v2, log |
  | `c7i.2xlarge-abhinav__…-Q8_0` | Baseline on a second instance (reference for the RAM rows) | v1 |
  | `c7i.2xlarge-amx__…-Q8_0`, `c7i.2xlarge-noamx__…-Q8_0` | AMX on and off, same instance | v1 |
  | `c6i.2xlarge__…-Q8_0` | Intel Ice Lake, no AMX | v1, v2, log |
  | `c7a.2xlarge__…-Q8_0` | AMD Genoa | v1, v2 (partial sweep), log |
  | `c8g.2xlarge__…-Q8_0` | ARM Graviton4 | v2 (partial sweep) |
  | `c7i.1xlarge-q8__…-Q8_0`, `c7i.4xlarge-q8__…-Q8_0` | 4 and 16 vCPUs (`c7i.1xlarge` is the c7i.xlarge) | v1 |
  | `m7i.2xlarge__…-Q8_0`, `r7i.2xlarge__…-Q8_0` | 32 and 64 GB RAM | v1 |
  | `c7i.2xlarge__…-Q4_K_M` | 4-bit quantization; Q4_K_M concurrency sweep | v1, v2, log |
  | `c7i.2xlarge__…-f16` | 16-bit weights | v1, log |
  | `c7a.2xlarge-p2__…`, `c7i.xlarge-p2__…`, `m8g.xlarge-p2__…` | Routing cluster node profiles | v2 |

- [`results/routing/`](../results/routing/): [`summary.csv`](../results/routing/summary.csv) (one row per run and request group), [`requests.csv`](../results/routing/requests.csv) (one row per request), [`nodes.json`](../results/routing/nodes.json) (the cluster configuration used), [`c7i-memory-rerun.csv`](../results/routing/c7i-memory-rerun.csv) (c7i.xlarge memory samples during the rerun), and [`routing-console.log`](../results/routing/routing-console.log).
- [`results/quality/`](../results/quality/): [`answers.csv`](../results/quality/answers.csv), [`scores.csv`](../results/quality/scores.csv), [`perplexity.csv`](../results/quality/perplexity.csv), and the raw perplexity logs (`perplexity-<quant>.log`).

## 4. Results Format

Each row in the v2 files records its configuration (node, model, slots, concurrency, output mode, and `run_id`), so rows can be filtered and compared across folders; the v1 files record node, model, and concurrency. In the baseline results, `summary*.csv` reports medians per workload (end-to-end latency and p95, TTFT, prefill and decode speed, queue wait, throughput, peak memory, and CPU use), and `requests*.csv` has the same measurements for every request. Compare end-to-end latency across nodes on `fixed` output rows, and match individual requests across nodes by `run`, `input`, `concurrency`, and `output_mode`. The routing results report the same latencies per run, for all requests, each request type, and each node, plus cost per 1,000 requests. Full metric definitions: [`docs/baseline-benchmark.md`](baseline-benchmark.md) (section 4) and [`docs/routing-benchmark.md`](routing-benchmark.md) (section 5).
