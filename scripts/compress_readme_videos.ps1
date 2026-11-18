param([string]$FFmpeg = '')

# 전체 길이와 해상도를 보존하면서 GitHub 첨부용 10 MB 미만 영상으로 압축한다.
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
if (-not $FFmpeg) {
    $bundled = Join-Path $root 'analysis_results/training_2000_tools/packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v9.0.2.exe'
    if (Test-Path -LiteralPath $bundled) { $FFmpeg = $bundled }
    elseif (Get-Command ffmpeg -ErrorAction SilentlyContinue) { $FFmpeg = (Get-Command ffmpeg).Source }
    else { throw 'FFmpeg를 설치하거나 -FFmpeg로 실행 파일 경로를 지정하세요.' }
}
$work = Join-Path $root 'analysis_results/readme_video_upload'
New-Item -ItemType Directory -Force -Path $work | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root 'docs/assets') | Out-Null
$rows = @()
foreach ($case in @(@('19953301', '1500k'), @('37277130', '2600k'))) {
    $id = $case[0]
    $source = Join-Path $root ('analysis_results/public_video_comparison/pexels_' + $id + '_두_모델_비교.mp4')
    $target = Join-Path $root ('docs/assets/public-models-' + $id + '.mp4')
    $passLog = Join-Path $work ('encode-' + $id)
    $common = @('-y','-hide_banner','-loglevel','error','-i',$source,'-an','-vf','fps=15',
        '-c:v','libx264','-preset','fast','-b:v',$case[1],'-pix_fmt','yuv420p','-passlogfile',$passLog)
    Write-Output "비교 영상 $id 압축: 1/2 단계"
    & $FFmpeg @common -pass 1 -f null NUL
    if ($LASTEXITCODE -ne 0) { throw "영상 $id 압축 1단계 실패" }
    Write-Output "비교 영상 $id 압축: 2/2 단계"
    & $FFmpeg @common -pass 2 -movflags +faststart $target
    if ($LASTEXITCODE -ne 0) { throw "영상 $id 압축 2단계 실패" }
    $file = Get-Item -LiteralPath $target
    if ($file.Length -ge 10000000) { throw "영상 $id 파일이 GitHub 첨부 제한 10 MB 이상입니다." }
    $rows += [ordered]@{
        id=$id; source=('analysis_results/public_video_comparison/pexels_' + $id + '_두_모델_비교.mp4')
        output=('docs/assets/public-models-' + $id + '.mp4'); bytes=$file.Length; fps=15; bitrate=$case[1]
        sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        operation='전체 길이·해상도 유지, H.264 2-pass 고정 비트레이트, 15fps, 음성 없음'
    }
    Write-Output ("비교 영상 $id 완료: " + [math]::Round($file.Length / 1000000, 2) + ' MB')
}
$rows | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $work 'compression.json') -Encoding utf8
