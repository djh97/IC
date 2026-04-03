param(
    [string]$Purpose = "implementation_bootstrap",
    [string]$SourceDir = "",
    [string]$TemplateFile = "",
    [string]$PatientProfileFile = ""
)

$cmd = @("run_pipeline.py", "init-run", "--purpose", $Purpose)

if ($SourceDir -ne "") {
    $cmd += @("--source-dir", $SourceDir)
}

if ($TemplateFile -ne "") {
    $cmd += @("--template-file", $TemplateFile)
}

if ($PatientProfileFile -ne "") {
    $cmd += @("--patient-profile-file", $PatientProfileFile)
}

& python $cmd
