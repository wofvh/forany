
$csvFilePath = "./hashes.csv"

"File,SHA256Hash" | Out-File -FilePath $csvFilePath -Encoding UTF8

Get-ChildItem -Recurse | ForEach-Object {
    $hash = Get-FileHash $_.FullName -Algorithm SHA256
    "$($_.FullName),$($hash.Hash)" | Out-File -FilePath $csvFilePath -Append -Encoding UTF8
}

Write-Host "Hash values have been saved to $csvFilePath"
