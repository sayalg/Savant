#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

// ─────────────────────────────────────────────────────────────────────────────
//  PARAMETERS  — mirror your existing sbatch arguments
// ─────────────────────────────────────────────────────────────────────────────
params.input_csv         = "jobs.csv"
params.outdir            = "results"
params.pipeline_script   = "scripts/pipeline.py"
params.singularity_image = "/scratch/sguirale/AlphaFold3/image/alphafold3.sif"
params.model_params      = "/projects/datasets/alphafold/3.0.0/weights"
params.databases         = "/scratch/sguirale/AlphaFold3/databases"
params.code_dir          = "/scratch/sguirale/AlphaFold3/code"
params.max_gpus          = 20
params.max_retries       = 2

// ─────────────────────────────────────────────────────────────────────────────
//  INPUT CHANNEL — one tuple per row in the CSV
// ─────────────────────────────────────────────────────────────────────────────
Channel
    .fromPath(params.input_csv)
    .splitCsv(header: true, strip: true)
    .map { row -> tuple(row.job_name, row.sequence, row.smiles) }
    .set { jobs_ch }

// ─────────────────────────────────────────────────────────────────────────────
//  PROCESS — one GPU job per CSV row, exactly mirroring your test command
// ─────────────────────────────────────────────────────────────────────────────
process DOCK {
    tag          "$job_name"
    label        'gpu'
    maxForks     params.max_gpus   // never exceed GPU cap cluster-wide
    maxRetries   params.max_retries
    errorStrategy { task.attempt <= params.max_retries ? 'retry' : 'ignore' }

    publishDir "${params.outdir}/${job_name}", mode: 'copy', overwrite: true

    input:
        tuple val(job_name), val(sequence), val(smiles)

    output:
        // The output dir your pipeline.py creates, named by -n argument
        tuple val(job_name), path("${job_name}/")

    script:
    """
    python3 ${params.pipeline_script} \\
        -p '${sequence}' \\
        -s '${smiles}' \\
        --singularity-image ${params.singularity_image} \\
        --model-params      ${params.model_params} \\
        --databases         ${params.databases} \\
        --code-dir          ${params.code_dir} \\
        -n                  ${job_name}
    """
}

// ─────────────────────────────────────────────────────────────────────────────
//  PROCESS — write one status row per job
//  Captures both successful and failed jobs by joining against the input channel
// ─────────────────────────────────────────────────────────────────────────────
process WRITE_STATUS {
    tag          "$job_name"
    label        'cpu_small'
    errorStrategy 'ignore'

    input:
        tuple val(job_name), val(status)

    output:
        path "${job_name}.status.csv"

    script:
    """
    echo "${job_name},${status}" > ${job_name}.status.csv
    """
}

// ─────────────────────────────────────────────────────────────────────────────
//  PROCESS — merge all status rows into a single manifest
// ─────────────────────────────────────────────────────────────────────────────
process MANIFEST {
    label      'cpu_small'
    publishDir params.outdir, mode: 'copy', overwrite: true

    input:
        path status_files   // collects all *.status.csv as a list

    output:
        path "run_manifest.csv"

    script:
    """
    echo "job_name,status" > run_manifest.csv
    cat ${status_files} >> run_manifest.csv
    echo "Manifest written: \$(wc -l < run_manifest.csv) rows"
    """
}

// ─────────────────────────────────────────────────────────────────────────────
//  WORKFLOW
// ─────────────────────────────────────────────────────────────────────────────
workflow {

    // Run docking
    dock_out = DOCK(jobs_ch)

    // Build a (job_name, status) channel covering both outcomes:
    //   SUCCESS — job_name appears in DOCK's output channel
    //   FAILED  — job_name is in the input but absent from DOCK's output
    successful_names = dock_out.map { job_name, dir -> tuple(job_name, 'SUCCESS') }
    all_names        = jobs_ch.map  { job_name, seq, smi -> job_name }

    status_ch = all_names
        .join(successful_names, remainder: true)
        .map { job_name, status ->
            // null means DOCK produced no output for this job → FAILED
            tuple(job_name, status ?: 'FAILED')
        }

    // Write per-job status rows and collect into manifest
    rows = WRITE_STATUS(status_ch)
    MANIFEST(rows.collect())
}
