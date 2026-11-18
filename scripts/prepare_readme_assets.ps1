param([string]$FFmpeg = '', [switch]$ReuseVideos)

# 기존 실험의 미디어와 집계 결과를 README용 경로에 모은다. 추론·학습은 실행하지 않는다.
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$assets = Join-Path $root 'docs/assets'
$results = Join-Path $root 'docs/results'
New-Item -ItemType Directory -Force -Path $assets, $results | Out-Null
if (-not $FFmpeg) {
    $bundled = Join-Path $root 'analysis_results/training_2000_tools/packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v9.0.2.exe'
    if (Test-Path -LiteralPath $bundled) { $FFmpeg = $bundled }
    elseif (Get-Command ffmpeg -ErrorAction SilentlyContinue) { $FFmpeg = (Get-Command ffmpeg).Source }
}
$manifest = [System.Collections.Generic.List[object]]::new()
function Add-Record($source, $target, $description, $operation) {
    $manifest.Add([ordered]@{
        source = $source; output = $target; description = $description; operation = $operation
        source_sha256 = (Get-FileHash -LiteralPath (Join-Path $root $source) -Algorithm SHA256).Hash.ToLowerInvariant()
        output_sha256 = (Get-FileHash -LiteralPath (Join-Path $root $target) -Algorithm SHA256).Hash.ToLowerInvariant()
    })
}

$images = @(
    @('analysis_results/yolo26_seadronessee_1280_gpu/compare_1018_full_and_zoom.jpg', 'public-models-overview.jpg', 'SeaDronesSee 이미지 1018: 원본 전체 화면과 SSPNet·YOLO26m 확대 비교'),
    @('analysis_results/threshold_comparison/comparison_526.jpg', 'small-person-thresholds.jpg', 'SeaDronesSee 이미지 526: 작은 사람에 대한 두 모델·두 신뢰도 비교'),
    @('analysis_results/annotated_examples/images/piri_604_public_full.png', 'beach-public.png', 'AnnotationPiriapolis 이미지 604: 공개 SSPNet, 전체 이미지 추론'),
    @('analysis_results/annotated_examples/images/piri_604_finetuned_full.png', 'beach-finetuned.png', 'AnnotationPiriapolis 이미지 604: 5 epoch 추가 학습 SSPNet, 전체 이미지 추론'),
    @('analysis_results/annotated_examples/images/piri_604_public_zoom.png', 'beach-public-zoom.png', 'AnnotationPiriapolis 이미지 604: 공개 SSPNet 확대'),
    @('analysis_results/annotated_examples/images/piri_604_finetuned_zoom.png', 'beach-finetuned-zoom.png', 'AnnotationPiriapolis 이미지 604: 추가 학습 SSPNet 확대'),
    @('analysis_results/annotated_examples/images/sea_1018_public_zoom.png', 'external-public.png', 'SeaDronesSee 이미지 1018: 공개 SSPNet 확대'),
    @('analysis_results/annotated_examples/images/sea_1018_finetuned_zoom.png', 'external-finetuned.png', 'SeaDronesSee 이미지 1018: 5 epoch 추가 학습 SSPNet 확대')
)
foreach ($item in $images) {
    $target = 'docs/assets/' + $item[1]
    Copy-Item -LiteralPath (Join-Path $root $item[0]) -Destination (Join-Path $root $target)
    Add-Record $item[0] $target $item[2] '기존 이미지 복제'
}

foreach ($id in @('19953301', '37277130')) {
    $source = 'analysis_results/public_video_comparison/pexels_' + $id + '_두_모델_비교.mp4'
    $target = 'docs/assets/public-models-' + $id + '.mp4'
    $src = Join-Path $root $source
    $dst = Join-Path $root $target
    $poster = 'docs/assets/public-models-' + $id + '-poster.jpg'
    if ($ReuseVideos) {
        if (-not (Test-Path -LiteralPath $dst) -or -not (Test-Path -LiteralPath (Join-Path $root $poster))) {
            throw '재사용할 영상 또는 미리보기가 없습니다. -ReuseVideos 없이 실행하세요.'
        }
        $operation = '전체 길이·원래 해상도·프레임률 유지, H.264 CRF 25 재인코딩, 음성 없음'
        Add-Record $source $poster "Pexels $id 비교 영상의 첫 프레임" '영상 첫 프레임 추출'
    } elseif ($FFmpeg) {
        & $FFmpeg -y -hide_banner -loglevel error -i $src -an -c:v libx264 -preset fast -crf 25 -pix_fmt yuv420p -movflags +faststart $dst
        if ($LASTEXITCODE -ne 0) { throw "비교 영상 인코딩 실패: $source" }
        $operation = '전체 길이·원래 해상도·프레임률 유지, H.264 CRF 25 재인코딩, 음성 없음'
        & $FFmpeg -y -hide_banner -loglevel error -i $dst -frames:v 1 -q:v 2 (Join-Path $root $poster)
        if ($LASTEXITCODE -ne 0) { throw "영상 미리보기 추출 실패: $source" }
        Add-Record $source $poster "Pexels $id 비교 영상의 첫 프레임" '영상 첫 프레임 추출'
    } else {
        throw '영상과 미리보기 생성에 FFmpeg가 필요합니다. -FFmpeg로 실행 파일 경로를 지정하세요.'
    }
    Add-Record $source $target "Pexels $id 공개 모델 비교: 왼쪽 YOLO26m, 오른쪽 SSPNet, 신뢰도 0.25" $operation
}

$metricFiles = @(
    @('analysis_results/seadronessee_validation/public/full/metrics_full.json', 'sea-public-025.json'),
    @('analysis_results/seadronessee_validation/public/full/metrics_score_005.json', 'sea-public-005.json'),
    @('analysis_results/seadronessee_validation/finetuned/full/metrics_full.json', 'sea-finetuned-025.json'),
    @('analysis_results/yolo26_seadronessee_1280_gpu/metrics.json', 'sea-yolo26m-025.json'),
    @('analysis_results/yolo26_seadronessee_1280_gpu_005/metrics.json', 'sea-yolo26m-005.json'),
    @('analysis_results/tiles_comparison_20261111/full_all_people.json', 'beach-public-full.json'),
    @('analysis_results/tiles_comparison_20261111/tiles_all_people.json', 'beach-public-tiles.json')
)
foreach ($item in $metricFiles) {
    $target = 'docs/results/' + $item[1]
    Copy-Item -LiteralPath (Join-Path $root $item[0]) -Destination (Join-Path $root $target)
    Add-Record $item[0] $target '저장된 평가 지표' '기존 JSON 복제'
}
foreach ($mode in @('full', 'tiles')) {
    $source = "analysis_results/local_gpu_smoke/evaluation_${mode}_all/result.json"
    $j = Get-Content -LiteralPath (Join-Path $root $source) -Raw -Encoding UTF8 | ConvertFrom-Json
    $target = "docs/results/beach-finetuned-$mode.json"
    $j.all_people | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $root $target) -Encoding utf8
    Add-Record $source $target 'AnnotationPiriapolis 추가 학습 모델의 전체 사람 평가' 'result.json의 all_people 추출'
}
$waterSources = @(
    @('analysis_results/tiles_comparison_20261111/full_water_only.json', 'public_full'),
    @('analysis_results/tiles_comparison_20261111/tiles_water_only.json', 'public_tiles'),
    @('analysis_results/local_gpu_smoke/evaluation_full_all/result.json', 'finetuned_full'),
    @('analysis_results/local_gpu_smoke/evaluation_tiles_all/result.json', 'finetuned_tiles')
)
$water = [ordered]@{}
foreach ($item in $waterSources) {
    $value = Get-Content -LiteralPath (Join-Path $root $item[0]) -Raw -Encoding UTF8 | ConvertFrom-Json
    $water[$item[1]] = if ($item[1].StartsWith('finetuned')) { $value.water_only } else { $value }
}
$waterTarget = 'docs/results/beach-water-only.json'
$water | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $root $waterTarget) -Encoding utf8
foreach ($item in $waterSources) {
    Add-Record $item[0] $waterTarget ("물속 사람 보조 평가: " + $item[1]) '기존 집계 네 조합을 병합, result.json에서는 water_only 추출'
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $results 'provenance.json') -Encoding utf8
Write-Output "README용 미디어·결과 준비 완료: $($manifest.Count)개"
