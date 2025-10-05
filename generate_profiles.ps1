# Specify the directory path
$directoryBenchmark = "path\to\benchmark"
$directoryStoreProfiles = "path\to\store\profiles"

$files = Get-ChildItem -Path $directoryBenchmark -File

# Create directoy to store profiles if it does not exist
if (-not (Test-Path -Path $directoryStoreProfiles)) {
    New-Item -Path $directoryStoreProfiles -ItemType Directory
    Write-Host "Directory created: $directoryStoreProfiles"
}

# Define the script block to be executed in parallel
$block = {
    Param([string] $file, [string] $directoryStoreProfiles)
    & "C:\Java\jdk-21.0.1\bin\java.exe" -jar "path\to\FREYJA-all.jar" "createProfile" $file $directoryStoreProfiles "false"
}

# Measure the time taken to execute the loop
$executionTime = Measure-Command {
    #Remove all jobs
    Get-Job | Remove-Job

    # Define the maximum number of concurrent jobs
    $MaxThreads = 8

    # Start the jobs. Max 8 jobs running simultaneously.
    for ($i = 0; $i -lt $files.Count; $i++) {
        While ($(Get-Job -state running).count -ge $MaxThreads){
            Start-Sleep -Milliseconds 10
        }
        Write-Host "Processing file $i /"$files.Count ":" $files[$i].FullName
        Start-Job -Scriptblock $block -ArgumentList $files[$i].FullName, $directoryStoreProfiles
    }

    # Wait for all jobs to finish.
    While ($(Get-Job -State Running).count -gt 0){
        Start-Sleep 1
    }

    # Remove all jobs created.
    Get-Job | Remove-Job
}

Write-Host "Loop execution time: $($executionTime.TotalSeconds) seconds"
