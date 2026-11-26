# out/eval의 이미지로 정적 갤러리(index.html) 생성. 같은 프롬프트/seed의 LoRA 없음/있음을 한 줄에 나란히.
# 사용: python3 -I tools/gallery.py [폴더=out/eval]
# 보기: python3 -m http.server 8000 --bind 127.0.0.1 --directory out/eval  (+ cloudflared 빠른 터널)
import html
import os
import re
import sys

D = sys.argv[1] if len(sys.argv) > 1 else 'out/eval'
files = sorted(f for f in os.listdir(D) if f.endswith('.png'))

PROMPTS = {
    'p0': 'a photo of sks dog',
    'p1': 'a photo of sks dog in a bucket',
    'beach': 'a photo of sks dog on a sunny beach',
    'snow': 'a photo of sks dog in a snowy forest',
}

# 행 키 = (프롬프트, 크기, seed), 열 = 설정
rows = {}
for f in files:
    m = re.match(r'base_(p\d+)_(\d+)_s(\d+)\.png$', f)
    if m:
        rows.setdefault((m[1], m[2], m[3]), []).append(('LoRA 없음', f))
        continue
    m = re.match(r'(dog_lora(?:_s\d+)?)_(p\d+)_lora_(\d+)_s(\d+)\.png$', f)
    if m:
        step = re.search(r'_s(\d+)$', m[1])
        rows.setdefault((m[2], m[3], m[4]), []).append((f'LoRA {step[1] if step else "600"}스텝', f))
        continue
    m = re.match(r'([a-z]+)_base_(\d+)_s(\d+)\.png$', f)
    if m:
        rows.setdefault((m[1], m[2], m[3]), []).append(('LoRA 없음', f))
        continue
    m = re.match(r'([a-z]+)_(dog_lora\w*?)_w([\d.]+)_(\d+)_s(\d+)\.png$', f)
    if m:
        rows.setdefault((m[1], m[4], m[5]), []).append((f'LoRA 600스텝 가중치 {m[3]}', f))
        continue
    rows.setdefault(('기타', '', ''), []).append((f, f))


def order(label):
    if label == 'LoRA 없음':
        return (0, 0, '')
    m = re.search(r'(\d+)스텝', label)
    return (1, int(m[1]) if m else 9999, label)


out = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
       '<title>SDXL LoRA on Vulkan (Radeon 660M)</title>',
       '<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}'
       'h2{font-size:1.05em;margin:28px 0 8px}.row{display:flex;gap:10px;flex-wrap:wrap}'
       'figure{margin:0;width:min(360px,46vw)}img{width:100%;border-radius:6px;display:block}'
       'figcaption{font-size:.85em;color:#bbb;padding:4px 0}a{color:inherit}</style>',
       '<h1>SDXL LoRA — 660M iGPU Vulkan 학습 결과</h1>',
       '<p>학습: diffusers/dog-example 5장, 캡션 "a photo of sks dog", 384px 600스텝, f32. '
       '생성: sd.cpp Vulkan, 768px, 20스텝, cfg 7, 같은 seed로 LoRA 없음/있음 비교.</p>']
for (p, size, seed), items in sorted(rows.items()):
    title = PROMPTS.get(p, p)
    out.append(f'<h2>"{html.escape(title)}" — {size}px, seed {seed}</h2><div class="row">')
    for label, f in sorted(items, key=lambda x: order(x[0])):
        out.append(f'<figure><a href="{html.escape(f)}"><img loading="lazy" src="{html.escape(f)}"></a>'
                   f'<figcaption>{html.escape(label)}</figcaption></figure>')
    out.append('</div>')
open(os.path.join(D, 'index.html'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print(f'{D}/index.html: {len(rows)}줄, 이미지 {len(files)}장')
