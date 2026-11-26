# 학습 데이터 캐시: 이미지 -> SDXL 잠재값, 캡션 -> crossattn/vector (.npy)
# sd.cpp DGL 스크립트를 이미지별로 펼쳐 생성한 뒤 sd-cli -M dgl로 한 번에 실행한다.
#
# 사용 (작업 폴더에서):
#   python3 -I tools/make_cache.py <이미지 폴더> <출력 폴더> [--size 512] [--caption "기본 캡션"] [--prefix "트리거, "]
#   이미지와 같은 이름의 .txt가 있으면 그 내용을 캡션으로 쓴다. 이미지는 가운데를 정사각형으로 잘라 size로 줄인다.
#   실행은 tools/guard.sh 아래에서: tools/guard.sh logs/cache.mem python3 -I tools/make_cache.py ...
import argparse
import json
import os
import subprocess
import sys

EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('dst')
ap.add_argument('--size', type=int, default=512)
ap.add_argument('--caption', default='')
ap.add_argument('--prefix', default='')
ap.add_argument('--model', default='models/sd_xl_base_1.0.safetensors')
ap.add_argument('--vae', default='models/sdxl_vae.safetensors')
ap.add_argument('--vae-backend', default='', help='비우면 기본(Vulkan), "cpu" 가능')
ap.add_argument('--sd-cli', default='sd-cli', help='sd.cpp의 sd-cli 실행 파일 (DGL 모드 지원 버전)')
ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()

assert a.size % 64 == 0, 'size는 64의 배수'
os.makedirs(a.dst, exist_ok=True)
items = []
for fn in sorted(os.listdir(a.src)):
    base, ext = os.path.splitext(fn)
    if ext.lower() not in EXTS:
        continue
    cap = a.caption
    txt = os.path.join(a.src, base + '.txt')
    if os.path.exists(txt):
        cap = open(txt, encoding='utf-8').read().strip()
    cap = a.prefix + cap
    name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in base)
    items.append((os.path.join(a.src, fn), cap, name))
if not items:
    sys.exit(f'이미지 없음: {a.src}')

q = json.dumps  # DGL 문자열 리터럴은 JSON과 같은 따옴표/이스케이프를 쓴다고 가정
lines = [
    '# version: 0.0.0',
    '# tools/make_cache.py가 생성 (손으로 고치지 말 것)',
    '',
    'fn prep(path, size) {',
    '    img = image::load(path)',
    '    w = tensor::shape(img)[1]',
    '    h = tensor::shape(img)[2]',
    '    side = math::min(w, h)',
    '    x0 = math::floor(math::div(math::sub(w, side), 2))',
    '    y0 = math::floor(math::div(math::sub(h, side), 2))',
    '    sq = tensor::crop(img, axes = [1, 2], start = [x0, y0], size = [side, side])',
    '    return tensor::interpolate(sq, axes = [1, 2], size = [size, size], mode = "bicubic", antialias = true)',
    '}',
    '',
    'fn main() {',
    f'    text = load_text_encoder({q(a.model)}, family = "sdxl", backend = "cpu")',
    f'    vae = load_vae({q(a.vae)}, family = "sdxl", conv_scale = 0.03125, flash_attn_mode = "off"'
    + (f', backend = {q(a.vae_backend)}' if a.vae_backend else '') + ')',
    '    rng = make_rng("cuda", seed = 0)',
]
for i, (path, cap, name) in enumerate(items):
    out = os.path.join(a.dst, name)
    lines += [
        f'    px{i} = prep({q(path)}, {a.size})',
        f'    image::save({q(out + ".check.png")}, px{i})',
        f'    lat{i} = vae_to_unet_latents(vae, vae_encode(vae, image::to_f32(px{i}, channels = 3), rng = rng, posterior = "mode"))',
        f'    tensor::save({q(out + ".latent.npy")}, lat{i})',
        f'    c{i} = encode_prompt(text, {q(cap)}, width = {a.size}, height = {a.size})',
        f'    tensor::save({q(out + ".crossattn.npy")}, c{i}.crossattn)',
        f'    tensor::save({q(out + ".vector.npy")}, c{i}.vector)',
    ]
lines += ['    return;', '}', '']
script = os.path.join(a.dst, 'cache.dgl')
open(script, 'w').write('\n'.join(lines))
with open(os.path.join(a.dst, 'captions.txt'), 'w', encoding='utf-8') as f:
    for path, cap, name in items:
        f.write(f'{name}\t{path}\t{cap}\n')
print(f'{len(items)}개 이미지, 스크립트: {script}')

r = subprocess.run([a.sd_cli, '-M', 'dgl', '--script', script, '--check'], capture_output=True, text=True)
print((r.stdout + r.stderr).strip().splitlines()[-1])
if r.returncode != 0 or a.dry_run:
    sys.exit(r.returncode)
r = subprocess.run([a.sd_cli, '-M', 'dgl', '--script', script, '-t', '10'], capture_output=True, text=True)
log = r.stdout + r.stderr
open(os.path.join(a.dst, 'cache.log'), 'w').write(log)
for ln in log.splitlines():
    if ln.startswith('[E]') or ln.startswith('[W]') or 'completed' in ln:
        print(ln[:200])
sys.exit(r.returncode)
