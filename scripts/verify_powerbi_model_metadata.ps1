<#: Validate model.bim using installed Power BI TOM libraries without controlling Desktop
    or connecting to its live Analysis Services instance. Does not execute M or DAX. #>
param(
    [Parameter(Mandatory=$true)][string]$PowerBIBin,
    [string]$ModelPath = (Join-Path $PSScriptRoot '../dashboard/powerbi/LoanRisk.SemanticModel/model.bim'),
    [string]$ResultPath = (Join-Path $PSScriptRoot '../dashboard/powerbi/model_metadata_verification.json')
)
$ErrorActionPreference = 'Stop'
foreach ($loanLibrary in @('Microsoft.AnalysisServices.Server.Core.dll',
    'Microsoft.AnalysisServices.Server.Tabular.dll','Microsoft.AnalysisServices.Server.Tabular.Json.dll')) {
    [void][System.Reflection.Assembly]::LoadFrom((Join-Path $PowerBIBin $loanLibrary))
}
$loanText = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $ModelPath).Path)
$loanDatabase = [Microsoft.AnalysisServices.Tabular.JsonSerializer]::DeserializeDatabase($loanText)
$loanValidation = $loanDatabase.Model.Validate()
$loanErrors = @($loanValidation.Errors | ForEach-Object { $_.ToString() })
$loanResult = [ordered]@{
    status = $(if ($loanValidation.ContainsErrors) {'failed'} else {'passed'})
    validation_scope = 'installed_TOM_deserialization_and_model_metadata_validation_only'
    desktop_library_version = (Get-Item -LiteralPath (Join-Path $PowerBIBin 'PBIDesktop.exe')).VersionInfo.FileVersion
    model_sha256 = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    compatibility_level = $loanDatabase.CompatibilityLevel
    declared_data_source_version = ($loanText | ConvertFrom-Json).model.defaultPowerBIDataSourceVersion
    tables = $loanDatabase.Model.Tables.Count
    errors = $loanErrors
    desktop_open_verified = $false
    dax_engine_execution_verified = $false
    power_query_execution_verified = $false
}
$loanJson = $loanResult | ConvertTo-Json -Depth 5
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($ResultPath),$loanJson,[System.Text.UTF8Encoding]::new($false))
Write-Output $loanJson
if ($loanValidation.ContainsErrors) { throw 'Power BI TOM model metadata validation failed' }
